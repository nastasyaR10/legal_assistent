"""Ядро ассистента: answer_question(messages) -> Answer.

Цепочка: понять вопрос -> найти фрагменты -> собрать контекст -> сгенерировать ответ -> проверить.
Этот модуль вызывают solution.py (сабмит), бэкенд (веб-чат) и скрипт оценки.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import threading

import config
import llm_client
import prompts
from kb_loader import KnowledgeBase, load_kb, norm_number
from llm_client import log
from retriever import Retriever
from schemas import Answer, Message, ParsedQuery, Source

KB: KnowledgeBase = None
RETRIEVER: Retriever = None
_init_lock = threading.Lock()


# ================================================================ инициализация
def init_core(data_dir=None) -> None:
    global KB, RETRIEVER
    with _init_lock:
        if KB is None:
            KB = load_kb(data_dir)
            RETRIEVER = Retriever(KB)


def list_documents() -> list:
    init_core()
    return [{"doc_id": d.doc_id, "title": d.title, "doc_type": d.doc_type, "status": d.status}
            for d in KB.documents.values()]


def get_document(doc_id: str) -> dict:
    init_core()
    d = KB.documents[doc_id]
    return {"doc_id": d.doc_id, "title": d.title, "doc_type": d.doc_type, "status": d.status,
            "sections": [{"chunk_id": c.chunk_id, "section": c.section, "text": c.text}
                         for c in KB.chunks_of(doc_id)]}


# ================================================================ разбор диалога
ROLE_RE = re.compile(r"^\s*(Пользователь|Ассистент|User|Assistant)\s*:\s*", re.I | re.M)


def messages_from_text(raw: str) -> list:
    """Строка из questions.csv -> список реплик. Без префиксов — один вопрос пользователя."""
    raw = (raw or "").strip()
    marks = list(ROLE_RE.finditer(raw))
    if not marks:
        return [Message("user", raw)]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        role = "user" if m.group(1).lower() in ("пользователь", "user") else "assistant"
        content = raw[m.end():end].strip()
        if content:
            out.append(Message(role, content))
    if not out or out[-1].role != "user":
        return [Message("user", raw)]
    return out


# ================================================================ понимание вопроса
def _stem_word(w: str) -> str:
    w = w.lower().replace("ё", "е")
    return re.sub(r"(ый|ий|ой|ая|ое|ые|ого|а|я|о|е|ы|и|й)$", "", w) if len(w) > 4 else w


def _tokens(text: str) -> list:
    return re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))


def _name_matches(name_stems: list, toks: list) -> bool:
    n = len(name_stems)
    for i in range(len(toks) - n + 1):
        if all(toks[i + j].startswith(name_stems[j]) for j in range(n)):
            return True
    return False


def find_contracts(text: str) -> tuple:
    """-> (найденные doc_id, неоднозначные кандидаты)."""
    found, toks = [], _tokens(text)
    # 1) номер договора
    for m in re.finditer(r"([A-ZА-ЯЁa-zа-яё]{1,3})\s*-?\s*(\d{1,3})\s*[/\-]\s*(\d{4})", text):
        key = norm_number(m.group(0))
        for d in KB.documents.values():
            if d.number and norm_number(d.number) == key and d.doc_id not in found:
                found.append(d.doc_id)
    # 2) контрагент; более длинное имя проверяем первым, короткое — только если токен не продолжает длинное
    cands = []
    for d in KB.contracts:
        if d.counterparty:
            name = re.sub(r"^ООО\s*|[«»\"]", "", d.counterparty)
            stems = [_stem_word(w) for w in re.findall(r"[A-Za-zА-Яа-яЁё]+", name)]
            cands.append((d, stems, [w.lower().replace("ё", "е") for w in re.findall(r"[A-Za-zА-Яа-яЁё]+", name)]))
    full = []
    for d, stems, words in cands:
        if _name_matches(stems, toks):
            # если есть кандидат с более длинным последним словом, который тоже совпал, — он точнее
            longer = [o for o, os_, ow in cands if o is not d and len(ow) == len(words)
                      and ow[-1].startswith(words[-1]) and len(ow[-1]) > len(words[-1]) and _name_matches(ow, toks)]
            if not longer:
                full.append(d.doc_id)
    for did in full:
        if did not in found:
            found.append(did)
    # 3) неоднозначность: совпало только первое слово названия у нескольких договоров («Альфа»)
    ambiguous = []
    if not found:
        partial = [d for d, stems, _ in cands if len(stems) > 1 and any(t.startswith(stems[0]) for t in toks)]
        if len(partial) > 1:
            ambiguous = partial
    return found, ambiguous


MONTH_RE = r"(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*"


def find_event_date(text: str):
    t = text.lower()
    m = re.search(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,2}\s+)?" + MONTH_RE + r"\s+(\d{4})", t)
    if m:
        return m.group(0)
    m = re.search(r"\bв\s+(\d{4})\s+год", t) or re.search(r"\b(\d{4})\s+года\b", t)
    if m:
        return m.group(0)
    return None


ABBREVIATIONS = {
    "НДС": "налог на добавленную стоимость налоговая ставка",
    "НДФЛ": "налог на доходы физических лиц налоговая ставка",
    "МРОТ": "минимальный размер оплаты труда",
    "ПДн": "персональные данные",
    "ГД": "генеральный директор",
    "СЭД": "система электронного документооборота",
    "УПД": "универсальный передаточный документ",
    "ЮО": "юридический отдел",
    "юротдел\\w*": "юридический отдел",
    "АХО": "административно-хозяйственный отдел",
}

def _expand_abbr(text: str) -> str:
    """Расшифровка аббревиатур — в документах они часто написаны полностью."""
    extra = [full for abbr, full in ABBREVIATIONS.items() if re.search(r"\b" + abbr + r"\b", text, re.I)]
    return text + ("\n" + " ".join(extra) if extra else "")


MONTH_NUM = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7, "август": 8,
             "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}


def event_date_to_date(phrase):
    """«августе 2024» -> (date(2024, 8, 15), «август 2024»); «в 2024 году» -> (date(2024, 7, 1), «2024 год»)."""
    if not phrase:
        return None
    t = phrase.lower()
    m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", t)
    if m:
        try:
            return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    m = re.search(r"(?:(\d{1,2})\s+)?(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+(\d{4})", t)
    if m:
        stem = m.group(2)
        month = next(v for k, v in MONTH_NUM.items() if stem.startswith(k))
        try:
            return dt.date(int(m.group(3)), month, int(m.group(1)) if m.group(1) else 15)
        except ValueError:
            return None
    m = re.search(r"(\d{4})", t)
    if m and 1990 < int(m.group(1)) < 2100:
        return dt.date(int(m.group(1)), 7, 1)
    return None


def _iso(s):
    try:
        return dt.date.fromisoformat(s) if s else None
    except ValueError:
        return None


def current_edition_hints(blocks: list, focus: set = None) -> list:
    """Даты события нет, а в контексте несколько редакций: прямо говорим, какая действует сейчас."""
    regs = []
    for b in blocks:
        d = KB.documents[b["doc_id"]]
        if focus is not None and d.doc_id not in focus:
            continue
        if d.doc_type == "regulation" and d not in regs:
            regs.append(d)
    if len(regs) < 2:
        return []
    active = [d for d in regs if d.status == "active"]
    expired = [d for d in regs if d.status == "expired"]
    if not active:
        return []
    out = [f"Дата события в вопросе не указана: отвечай по ДЕЙСТВУЮЩЕЙ редакции — {active[0].title} — и отметь в "
           f"ответе, что вывод дан по состоянию на текущую дату."]
    for d in expired:
        out.append(f"{d.title} утратила силу" + (f" с {d.valid_to}" if d.valid_to else "") +
                   " — применяй её только к событиям до этой даты, иначе упоминай лишь для сравнения.")
    return out


def edition_hints(q: ParsedQuery, blocks: list, top_docs=None, top_chunks=None) -> list:
    """Подсказки модели: какая редакция документа действовала на дату события (считается кодом, не моделью)."""
    when = event_date_to_date(q.event_date)
    if not when:
        return []
    hints, seen = [], set()
    focus = set(q.doc_ids) | set(top_docs or [])
    for did in list(focus):
        focus.update(KB.documents[did].related)
    for b in blocks:
        d = KB.documents[b["doc_id"]]
        if d.doc_id in seen or (focus and d.doc_id not in focus):
            continue
        seen.add(d.doc_id)
        if d.doc_type == "regulation" and (d.valid_from or d.valid_to):
            start, end = _iso(d.valid_from), _iso(d.valid_to)
            ok = (start is None or when >= start) and (end is None or when < end)
            period = (f"действовала с {d.valid_from} по {d.valid_to}" if d.valid_to
                      else f"действует с {d.valid_from}")
            if ok:
                hints.append(f"На дату события ({q.event_date}) действовал документ: {d.title} ({period}). "
                             f"Отвечай по нему.")
            else:
                hints.append(f"{d.title} ({period}) к событию «{q.event_date}» НЕ применяется — "
                             f"можно упомянуть только для сравнения.")
        if d.doc_type != "law":
            text = d.text if len(b["chunk_ids"]) > 1 else KB.by_id[b["chunk_ids"][0]].text
            for sent in re.split(r"(?<=[.!?»])\s+", text):
                m = re.search(r"\b(после|с|до|не позднее)\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4})", sent, re.I)
                if not m or not re.search(r"примен|обязан|вступа|действ|подписан", sent, re.I):
                    continue
                pivot = event_date_to_date(m.group(2))
                if not pivot or pivot == when:
                    continue
                rel = "РАНЬШЕ" if when < pivot else "ПОЗЖЕ"
                hints.append(f"В документе «{d.title}» сказано: «{sent.strip()[:220]}». Дата события ({q.event_date}) — "
                             f"{rel} даты {pivot.strftime('%d.%m.%Y')}; сделай вывод с учётом этого.")
        if d.doc_type == "law":
            law_text = "\n".join(KB.by_id[c].text for bb in blocks if bb["doc_id"] == d.doc_id for c in bb["chunk_ids"]
                                 if top_chunks is None or c in top_chunks)
            done = set()
            for m in re.finditer(r"((?:Пункт|Часть|Статья|Подпункт|Абзац)[^\]\n]{0,25}?)\s*изменен\w*\s+с\s+"
                                 r"(\d{1,2})\s+([а-я]+)\s+(\d{4})", law_text + "\n" + d.text[:3000]):
                changed = event_date_to_date(f"{m.group(2)} {m.group(3)} {m.group(4)}")
                what = m.group(1).strip()
                if changed and when < changed and what not in done:
                    done.add(what)
                    hints.append(f"{d.short_ref}: «{what}» в базе приведён(а) в редакции, действующей с "
                                 f"{changed.strftime('%d.%m.%Y')}. Событие «{q.event_date}» — раньше этой даты, а "
                                 f"прежней редакции в базе нет: не называй прежнее значение по памяти, скажи, что "
                                 f"подтвердить его по базе нельзя, и приведи действующее значение для сравнения.")
    return hints


def premise_hints(question: str, context: str) -> list:
    """Числа/суммы из вопроса, которых нет в документах, — возможная ложная посылка."""
    hints = []
    ctx = _numbers(context)
    for m in re.finditer(r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:млн|тыс|рубл|%|процент|дн|месяц|год|лет|час)\w*", question):
        raw = m.group(0).strip()
        n = re.sub(r"[\s\u00a0]", "", re.match(r"\d[\d\s\u00a0]*(?:[.,]\d+)?", raw).group(0)).replace(",", ".")
        if len(n.replace(".", "")) <= 1 or n in ctx:
            continue
        if re.fullmatch(r"(19|20)\d\d", n) and re.search(r"год|г\.", raw):
            continue  # год события — не посылка
        hints.append(f"Значение «{raw}» из вопроса в документах контекста не встречается. Если вопрос исходит из того, "
                     f"что оно установлено документом, — поправь пользователя, назови верное значение и условия его "
                     f"изменения, если они есть в документе. Если это данные самого пользователя (сумма сделки, дата) — "
                     f"используй их для сравнения с условиями документов.")
    return hints


THRESHOLD_RE = re.compile(r"(свыше|более|превышающ\w*|не менее|не более|менее|до)\s+(\d[\d\s\u00a0]*)\s*(?:\([^)]*\)\s*)?"
                          r"рубл", re.I)


def threshold_hints(question: str, blocks: list, focus: set = None) -> list:
    """Сумма из вопроса vs пороги «свыше N рублей» в документах компании: сравнение делает код, а не модель."""
    q = _expand_scaled(question)
    amounts = []
    for m in re.finditer(r"(\d[\d\s\u00a0]*(?:[.,]\d+)?)\s*(?:руб|р\.|₽)", q):
        v = float(re.sub(r"[\s\u00a0]", "", m.group(1)).replace(",", "."))
        if v >= 1000:
            amounts.append(v)
    if not amounts:
        return []
    amount = amounts[0]
    hints = []
    for b in blocks:
        d = KB.documents[b["doc_id"]]
        if d.doc_type == "law" or (focus is not None and d.doc_id not in focus):
            continue
        for cid in b["chunk_ids"]:
            c = KB.by_id[cid]
            for m in THRESHOLD_RE.finditer(c.text):
                limit = float(re.sub(r"[\s\u00a0]", "", m.group(2)))
                if limit < 1000:
                    continue
                word = m.group(1).lower()
                if word in ("свыше", "более") or word.startswith("превыша"):
                    hit = amount > limit
                elif word == "не менее":
                    hit = amount >= limit
                elif word in ("не более", "до"):
                    hit = amount <= limit
                else:  # менее
                    hit = amount < limit
                st = "РЕДАКЦИЯ УТРАТИЛА СИЛУ" if d.status == "expired" else "действующая редакция"
                sent = " ".join(c.text.split("\n")[-1].split())[:200]
                fmt = lambda v: f"{int(v):,}".replace(",", " ")
                verdict = "ПОПАДАЕТ под это условие" if hit else "НЕ попадает под это условие"
                hints.append(f"[{d.short_ref}; {st}; {c.section}] «{sent}» — сумма из вопроса {fmt(amount)} руб. "
                             f"{verdict} ({word} {fmt(limit)} руб.).")
    return hints[:8]


def missing_reference_hints(blocks: list, top_docs: set = None) -> list:
    """Норма отсылает к статьям того же закона, которых нет в базе, — модель должна об этом сказать."""
    hints, texts = [], {}
    for b in blocks:
        if top_docs is not None and b["doc_id"] not in top_docs:
            continue
        if KB.documents[b["doc_id"]].doc_type == "law":
            texts.setdefault(b["doc_id"], []).extend(KB.by_id[c].text for c in b["chunk_ids"])
    for doc_id, parts in texts.items():
        d = KB.documents[doc_id]
        code = d.doc_id.split("-st")[0]
        text = "\n".join(parts)
        refs = set()
        for m in re.finditer(r"(?:стать(?:и|ей|ям|ях|ю|я)|ст\.)\s+((?:\d+(?:\.\d+)?(?:\s*(?:,|и)\s*)?)+)"
                             r"(?=[^.;]{0,40}настоящ)", text):
            refs.update(re.findall(r"\d+(?:\.\d+)?", m.group(1)))
        for m in re.finditer(r"част\w*\s+\d+\s+стать\w*\s+(\d+(?:\.\d+)?)", text):
            refs.add(m.group(1))
        own = d.doc_id.split("-st")[-1].replace("-", ".")
        missing = sorted({r for r in refs if r != own and f"{code}-st{r.replace('.', '-')}" not in KB.documents},
                         key=lambda x: float(x.split(".")[0]))
        if missing:
            hints.append(f"{d.short_ref} ссылается на статьи {', '.join(missing)} того же акта — их текстов в базе "
                         f"знаний нет. Это НЕ повод отказываться от ответа: ответь по тексту {d.short_ref}, а если "
                         f"вывод зависит от этих статей — добавь оговорку, что их в базе нет и применимость "
                         f"нужно уточнить у юриста.")
    return hints


def analyze(messages: list) -> ParsedQuery:
    last = messages[-1].content
    history = messages[:-1]
    found, ambiguous = find_contracts(last)
    if not found:
        # договор мог быть назван раньше в диалоге
        for msg in reversed(history):
            f, _ = find_contracts(msg.content)
            if f:
                found = f
                ambiguous = []
                break
    q = ParsedQuery(last_user_message=last, history=history, search_query=last, doc_ids=found)

    low = last.lower()
    if re.search(r"регламент|согласов|одобрен|инициатор|финансов\w* директор|генеральн\w* директор", low):
        q.doc_ids += [d.doc_id for d in KB.documents.values() if d.doc_type == "regulation" and d.doc_id not in q.doc_ids]

    q.event_date = find_event_date(last) or next((find_event_date(m.content) for m in reversed(history)
                                                  if m.role == "user" and find_event_date(m.content)), None)

    # поисковый запрос: последняя реплика + контекст диалога + названия найденных договоров
    parts = [last]
    if history:
        prev_user = [m.content for m in history if m.role == "user"][-2:]
        parts += [p[:300] for p in prev_user]
    for did in found:
        d = KB.documents[did]
        parts.append(f"{d.title}")
    q.search_query = _expand_abbr(" \n".join(parts))

    if ambiguous and not found:
        opts = [{"label": f"{d.counterparty} (договор № {d.number})",
                 "message": f"Речь о договоре № {d.number} с {d.counterparty}"} for d in ambiguous]
        names = " и ".join(f"{d.counterparty} (договор № {d.number})" for d in ambiguous)
        q.needs_clarification = True
        q.clarify_question = (f"Уточните, пожалуйста, о каком договоре идёт речь: в базе есть договоры с "
                              f"похожими контрагентами — {names}.")
        q.clarify_options = opts
        q.ambiguous_ids = [d.doc_id for d in ambiguous]
        q.search_query = _expand_abbr(q.search_query + " \n" + " \n".join(d.title for d in ambiguous))
    return q


# ================================================================ контекст
def _doc_header(d) -> str:
    if d.status == "expired":
        st = "РЕДАКЦИЯ УТРАТИЛА СИЛУ" + (f" с {d.valid_to}" if d.valid_to else "")
        if d.valid_from:
            st += f"; действовала с {d.valid_from}"
    else:
        st = "действует" + (f" с {d.valid_from}" if d.valid_from else "")
    extra = ""
    if d.doc_type == "amendment" and d.amends:
        extra = f"; изменяет договор {KB.documents[d.amends].short_ref}"
    if d.date and d.doc_type in ("contract", "amendment"):
        extra += f"; дата документа {d.date}"
    return f"{d.title} [{st}{extra}]"


def build_context(q: ParsedQuery, hits: list) -> tuple:
    """Широкий контекст -> (текст, блоки [{doc_id, chunk_ids, text}]).
    База маленькая, поэтому модели отдаём: все документы компании целиком (сначала относящиеся к вопросу),
    до 3 статей законов целиком (огромные статьи НК — лучшими фрагментами с соседями)."""
    blocks, used = [], set()
    budget = config.CONTEXT_MAX_CHARS
    pool = list(hits) + [h for h in getattr(q, "_all_hits", []) if h.chunk.chunk_id not in {x.chunk.chunk_id for x in hits}]

    def add_block(did, chunk_ids, label):
        nonlocal budget
        d = KB.documents[did]
        body = "\n".join(KB.by_id[c].text for c in chunk_ids)
        if budget - len(body) < 0:
            return False
        blocks.append({"doc_id": did, "chunk_ids": list(chunk_ids), "text": f"{_doc_header(d)} — {label}:\n{body}"})
        budget -= len(body)
        used.add(did)
        return True

    def add_company(did):
        if did in used or KB.documents[did].doc_type == "law":
            return
        add_block(did, KB.documents[did].chunk_ids, "полный текст")
        for rel in KB.documents[did].related:
            if rel not in used:
                add_block(rel, KB.documents[rel].chunk_ids, "полный текст")

    # 1) документы компании, названные в вопросе или найденные поиском
    for did in q.doc_ids:
        add_company(did)
    for h in hits[:6]:
        add_company(h.chunk.doc_id)

    # 2) законы: до 3 статей в порядке релевантности
    law_order = []
    for h in pool:
        if KB.documents[h.chunk.doc_id].doc_type == "law" and h.chunk.doc_id not in law_order:
            law_order.append(h.chunk.doc_id)
    for did in law_order[:3]:
        d = KB.documents[did]
        size = sum(len(KB.by_id[c].text) for c in d.chunk_ids)
        if size <= 25000:
            add_block(did, d.chunk_ids, "полный текст статьи")
            continue
        # огромная статья: найденные фрагменты + соседние, по порядку
        orders = sorted({KB.by_id[h.chunk.chunk_id].order for h in pool if h.chunk.doc_id == did})[:8]
        idx = sorted({o + k for o in orders for k in (-1, 0, 1) if 0 <= o + k < len(d.chunk_ids)})
        add_block(did, [d.chunk_ids[i] for i in idx], "фрагменты статьи")

    # 3) остальные документы компании — в конце, для полноты (все вместе ~17 тыс. символов)
    for did, d in KB.documents.items():
        if d.doc_type != "law":
            add_company(did)

    text = "\n\n".join(f"[{i + 1}] {b['text']}" for i, b in enumerate(blocks))
    return text, blocks


# ================================================================ генерация и проверка
def _parse_json(raw: str) -> dict:
    """Ответ модели -> {"status", "answer", "used"}. Переживает кривой JSON (лишние скобки, точки, переносы)."""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        for candidate in (m.group(0), m.group(0).replace("\n", " ")):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and data.get("answer"):
                    return data
            except json.JSONDecodeError:
                pass
    # ручной разбор: текст между "answer": " и закрывающей кавычкой перед "used"/"status"/концом
    am = re.search(r'"answer"\s*:\s*"', s)
    if am:
        tail = s[am.end():]
        end = re.search(r'"\s*[.,]?\s*(?:,\s*)?(?:"(?:used|status)"|\}|$)', tail)
        answer = tail[:end.start()] if end else tail
        answer = answer.replace('\\"', '"').replace("\\n", " ").strip()
        status = re.search(r'"status"\s*:\s*"(\w+)"', s)
        used = re.search(r'"used"\s*:\s*\[([\d,\s]*)\]', s)
        return {"status": status.group(1) if status else "answer", "answer": answer,
                "used": [int(x) for x in re.findall(r"\d+", used.group(1))] if used else []}
    return {"status": "answer", "answer": s, "used": []}


NUM_RE = re.compile(r"\d[\d\s ]*(?:[.,]\d+)?")

_UNITS = {"один": 1, "одна": 1, "одну": 1, "одного": 1, "одной": 1, "одним": 1, "два": 2, "две": 2, "двух": 2,
          "три": 3, "трех": 3, "трёх": 3, "четыре": 4, "четырех": 4, "четырёх": 4, "пять": 5, "пяти": 5,
          "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9,
          "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11, "двенадцать": 12, "двенадцати": 12,
          "тринадцать": 13, "тринадцати": 13, "четырнадцать": 14, "четырнадцати": 14, "пятнадцать": 15,
          "пятнадцати": 15, "шестнадцать": 16, "шестнадцати": 16, "семнадцать": 17, "семнадцати": 17,
          "восемнадцать": 18, "восемнадцати": 18, "девятнадцать": 19, "девятнадцати": 19, "двадцать": 20,
          "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40, "сорока": 40, "пятьдесят": 50,
          "пятидесяти": 50, "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70, "семидесяти": 70,
          "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90, "сто": 100, "ста": 100,
          "двести": 200, "двухсот": 200, "триста": 300, "трехсот": 300, "трёхсот": 300, "четыреста": 400,
          "четырехсот": 400, "четырёхсот": 400, "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "шестисот": 600,
          "семьсот": 700, "семисот": 700, "восемьсот": 800, "восьмисот": 800, "девятьсот": 900, "девятисот": 900}


def words_to_numbers(text: str) -> set:
    """«от ста пятидесяти тысяч до трехсот тысяч» -> {150000, 300000}; «два миллиона» -> {2000000}."""
    out, total, group, active = set(), 0, 0, False
    for w in re.findall(r"[а-яё]+|\S", text.lower()):
        if w in _UNITS:
            group += _UNITS[w]
            active = True
        elif w.startswith("тысяч") and active:
            total += (group or 1) * 1000
            group = 0
        elif w.startswith("миллион") and active:
            total += (group or 1) * 1_000_000
            group = 0
        else:
            if active and (total or group):
                out.add(str(total + group))
            total, group, active = 0, 0, False
    if active and (total or group):
        out.add(str(total + group))
    return out



def _numbers(text: str) -> set:
    out = set(words_to_numbers(text))
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        n = re.sub(r"[\s ]", "", raw).replace(",", ".").rstrip(".")
        if n:
            out.add(n)
        for part in re.split(r"[\s ]+", raw):   # «до 5 рабочих» и «450 000» — оба варианта
            p = part.replace(",", ".").rstrip(".")
            if p:
                out.add(p)
    return out


def _expand_scaled(text: str) -> str:
    """«2,4 млн» -> «2400000», «312 тыс.» -> «312000», чтобы сравнивать с числами в документах."""
    def rep(m):
        val = float(m.group(1).replace(" ", "").replace("\u00a0", "").replace(",", "."))
        mult = 1_000_000 if m.group(2).startswith("млн") or m.group(2).startswith("миллион") else 1000
        v = val * mult
        return f"{int(v)}" if v == int(v) else m.group(0)
    return re.sub(r"(\d[\d\s\u00a0]*(?:[.,]\d+)?)\s*(млн|миллион\w*|тыс\.?|тысяч\w*)", rep, text)


DOCNUM_RE = re.compile(r"№\s*[\w.\-/]+|\b[А-ЯЁA-Z]{1,4}-\d+(?:/\d{2,4})?\b|\bст\.\s*\d+(?:\.\d+)*|"
                       r"\bп\.\s*\d+(?:\.\d+)*|\bч\.\s*\d+(?:\.\d+)*|\b\d{1,2}\.\d{1,2}\.\d{4}\b")


def _derived(nums: set, percents: set) -> set:
    """Простые производные чисел из документов и вопроса: p% от X, X±Y — чтобы расчёты не считались выдумкой."""
    big = sorted({float(n) for n in nums if re.fullmatch(r"\d+(?:\.\d+)?", n) and float(n) >= 100})[:60]
    out = set()
    def add(v):
        if v > 0 and abs(v - round(v)) < 1e-6:
            out.add(str(int(round(v))))
    for p in percents:
        for x in big:
            add(x * p / 100)
            for y in big:
                add(x * p / 100 + y)
    for x in big:
        for y in big:
            add(x + y)
            add(abs(x - y))
    for p in percents:            # (X − Y) × p% + Z: прогрессивные ставки
        for x in big:
            for y in big:
                if x > y:
                    for z in big[:30]:
                        add((x - y) * p / 100 + z)
    return out


def check_answer(answer: str, context: str) -> list:
    """Числа из ответа, которых нет в контексте (кроме однозначных, номеров документов и пунктов)."""
    ctx = _numbers(context) | _numbers(_expand_scaled(context))
    answer = DOCNUM_RE.sub(" ", _expand_scaled(answer))
    percents = {float(x.replace(",", ".")) for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:%|процент)", context + " " + answer)}
    percents |= {float(n) for n in words_to_numbers(" ".join(re.findall(r"([а-я]+\s+(?:процент\w*))", context)))}
    derived = None
    bad = []
    for m in NUM_RE.finditer(answer):
        n = re.sub(r"[\s ]", "", m.group(0)).replace(",", ".").rstrip(".")
        if not n or len(n.replace(".", "")) <= 1:
            continue
        parts = [x.replace(",", ".").rstrip(".") for x in re.split(r"[\s\u00a0]+", m.group(0).strip()) if x]
        if n in ctx or n.split(".")[0] in ctx or all(x in ctx or len(x) <= 1 for x in parts):
            continue
        if derived is None:
            derived = _derived(ctx, percents)
        if n in derived:
            continue
        bad.append(m.group(0).strip())
    return sorted(set(bad))


def _clean_text(text: str) -> str:
    t = re.sub(r"\*\*|__|^#+\s*", "", text, flags=re.M)
    repl = {"в": "в базе знаний", "из": "из базы знаний", "по": "по базе знаний", "": "в базе знаний"}
    t = re.sub(r"\b(?:(в|из|по)\s+)?(?:предоставленн|представленн)\w*\s+(?:мне\s+)?(?:контекст\w*|фрагмент\w*|"
               r"документ\w*)", lambda m: repl[(m.group(1) or "").lower()], t, flags=re.I)
    t = re.sub(r"(^|(?<!\bред)(?<!\bп)(?<!\bст)(?<!\bч)(?<!\bг)(?<!\bт)(?<!\bд)(?<!\bстр)(?<!\bабз)(?<!\bпп)"
               r"(?<!\bсм)(?<!\bруб)(?<!\bтыс)(?<!\bмлн)(?<!\bнаб)(?<!\bул)[.!?]\s+)([а-яё])",
               lambda m: m.group(1) + m.group(2).upper(), t)
    t = re.sub(r"\s*\[(?:фрагмент\w*\s*|блок\w*\s*)?\d+(?:\s*[,–-]\s*\d+)*\]", "", t, flags=re.I)
    t = re.sub(r"\s*\((?:фрагмент\w*|блок\w*)\s*\d+(?:\s*,\s*\d+)*\)", "", t, flags=re.I)
    t = re.sub(r'^\s*\{?\s*"?status"?\s*:\s*"\w+"\s*,\s*"?answer"?\s*:\s*"?', "", t)
    t = re.sub(r'"?\s*[.,]?\s*,?\s*"used"\s*:\s*\[[^\]]*\]\s*\}?\s*$', "", t)
    t = re.sub(r'"\s*\}\s*$', "", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t[:4000]


def fix_clause_numbers(text: str, blocks: list) -> tuple:
    """Если предложение со ссылкой «п. X.Y» по словам почти не совпадает с пунктом X.Y, но сильно совпадает
    с другим пунктом того же документа — исправляем номер. Возвращает (текст, список исправлений)."""
    from retriever import tokenize
    clauses = {}
    for b in blocks:
        chunks = [KB.by_id[c] for c in b["chunk_ids"]]
        for c in chunks:
            m = re.match(r"п\. (\d+\.\d+)$", c.section)
            if m:
                body = re.sub(r"^\d+\.\s+[^\n]*\n", "", c.text) if "\n" in c.text else c.text
                clauses.setdefault(m.group(1), []).append((c.doc_id, set(tokenize(body))))
    if not clauses:
        return text, []
    fixes = []
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])", text)
    for si, sent in enumerate(sentences):
        refs = re.findall(r"п\.\s*(\d+\.\d+)", sent)
        if len(refs) != 1 or refs[0] not in clauses:
            continue
        cited = refs[0]
        nums_in_sent = {norm_number(x) for x in re.findall(r"[А-ЯЁA-Z]{1,4}-\d+/\d{4}", sent)}
        words = set(tokenize(re.sub(r"\(.*?\)", "", sent)))
        if len(words) < 4:
            continue
        for doc_id, toks in clauses[cited]:
            dnum = KB.documents[doc_id].number
            if nums_in_sent and (not dnum or norm_number(dnum) not in nums_in_sent):
                continue  # предложение про другой договор — не трогаем
            own = len(words & toks)
            best_num, best = cited, own
            for num, lst in clauses.items():
                for d2, t2 in lst:
                    if d2 == doc_id and num != cited and len(words & t2) > best:
                        best_num, best = num, len(words & t2)
            if best_num != cited and best >= 3 and best >= 2 * max(own, 1):
                sentences[si] = re.sub(r"п\.\s*" + re.escape(cited) + r"\b", f"п. {best_num}", sent)
                fixes.append(f"{cited}->{best_num}")
            break
    return " ".join(sentences), fixes


def _sources(answer_text: str, blocks: list, used: list) -> list:
    out, seen = [], set()
    idx = [u - 1 for u in used if isinstance(u, int) and 0 < u <= len(blocks)] or list(range(min(2, len(blocks))))
    cited = set(re.findall(r"\b(\d+\.\d+)\b", answer_text))
    for i in idx:
        b = blocks[i]
        chunks = [KB.by_id[c] for c in b["chunk_ids"]]
        if len(chunks) > 1:   # полный документ: берём процитированные пункты
            picked = [c for c in chunks if c.section.replace("п. ", "") in cited] or chunks[:1]
        else:
            picked = chunks
        for c in picked:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            out.append(Source(chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title,
                              section=c.section, quote=c.text[:400], doc_status=c.doc_status))
    return out


def drop_sentences(text: str, problems: list) -> str:
    """Удаляет предложения, в которых встречаются неподтверждённые числа. Пустая строка — если не осталось ничего."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])", text)
    keep = [x for x in sentences if not any(p in x for p in problems)]
    if keep and len(keep) < len(sentences):
        return " ".join(keep).strip()
    # все предложения задеты — убираем только части предложений (между запятыми/точками с запятой)
    out = []
    for sent in sentences:
        parts = re.split(r"(?<=[,;])\s+", sent)
        good = [p_ for p_ in parts if not any(p in p_ for p in problems)]
        if good:
            joined = " ".join(good).rstrip(",; ")
            out.append(joined if joined.endswith((".", ")", "!", "?")) else joined + ".")
    res = " ".join(out).strip()
    return res if res and res != text else ""


CANNED_RE = re.compile(r"(не обладает собственным мнением|не обладают собственным мнением|разговоры на (?:некоторые|"
                       r"чувствительные) темы|временно ограничен|как и любая языковая модель|не могу обсуждать эту тему|"
                       r"I can(?:'|no)t help with)", re.I)
REFUSAL_RE = re.compile(r"(нет (?:информации|сведений|данных)|отсутству\w+ (?:информац|сведени|в (?:предоставленн|"
                        r"представленн|базе|контекст))|не могу (?:ответить|подтвердить)|не содерж\w+ (?:информац|"
                        r"сведени)|не (?:указан|установлен)\w* в (?:предоставленн|представленн)|в (?:предоставленном|"
                        r"представленном) контексте (?:нет|отсутству))", re.I)


def is_canned(raw: str) -> bool:
    """Пустой ответ или стандартная заглушка модели вместо ответа по документам."""
    return not (raw or "").strip() or bool(CANNED_RE.search(raw or ""))


def is_refusal(text: str, data: dict = None) -> bool:
    if data and data.get("status") == "refusal":
        return True
    return bool(REFUSAL_RE.search((text or "")[:300]))


def _relevant_found(q: ParsedQuery, hits: list) -> bool:
    """Есть основания считать, что ответ в найденном есть: назван документ или поиск уверенно нашёл фрагменты."""
    if q.doc_ids:
        return True
    if not hits:
        return False
    from retriever import tokenize
    qt = set(tokenize(q.last_user_message))
    best = max(len(qt & set(tokenize(h.chunk.text))) for h in hits[:4])
    return best >= 2


def _dialog_text(history: list) -> str:
    if not history:
        return ""
    lines = [("Пользователь: " if m.role == "user" else "Ассистент: ") + m.content for m in history[-6:]]
    return "ИСТОРИЯ ДИАЛОГА (для понимания контекста; факты бери только из КОНТЕКСТА):\n" + "\n".join(lines) + "\n\n"


def _search(q: ParsedQuery) -> list:
    """Основные результаты + до 4 фрагментов из других документов: названный договор и так попадёт в контекст
    целиком, а вопрос может требовать ещё и закона или регламента."""
    all_hits = RETRIEVER.search(q, top_n=config.FINAL_TOP + 16)
    q._all_hits = all_hits
    main = all_hits[:config.FINAL_TOP]
    if not q.doc_ids:
        return main
    targets = set(q.doc_ids)
    for did in q.doc_ids:
        targets.update(KB.documents[did].related)
    # отдельный поиск по самому вопросу вне названного договора (например, норма закона, на которую он опирается)
    plain = ParsedQuery(last_user_message=q.last_user_message, history=[], doc_ids=[],
                        search_query=_expand_abbr(q.last_user_message))
    try:
        extra = RETRIEVER.search(plain, top_n=4, exclude=targets, use_rerank=False)
    except Exception as exc:
        log(f"[core] дополнительный поиск не удался: {exc}")
        extra = [h for h in all_hits[config.FINAL_TOP:] if h.chunk.doc_id not in targets][:4]
    seen = {h.chunk.chunk_id for h in main}
    return main + [h for h in extra if h.chunk.chunk_id not in seen]


def answer_question(messages, fast: bool = False, allow_clarify: bool = False) -> Answer:
    """allow_clarify=True — для веб-чата: можно вернуть уточняющий вопрос. В сабмите уточнение засчитывается
    как отсутствие ответа, поэтому по умолчанию отвечаем по всем возможным вариантам."""
    if isinstance(messages, str):
        messages = messages_from_text(messages)
    # из веб-чата реплики приходят словарями {"role": ..., "content": ...}
    messages = [m if isinstance(m, Message) else Message(m.get("role", "user"), m.get("content", ""))
                for m in messages]
    init_core()
    q = analyze(messages)
    debug = {"doc_ids": q.doc_ids, "event_date": q.event_date, "prompt": prompts.PROMPT_VERSION}

    ambiguity_hint = None
    clarify_options = []
    if q.needs_clarification:
        if allow_clarify:
            return Answer(status="clarify", text=q.clarify_question, clarify_options=q.clarify_options,
                          debug={**debug, "question_type": "clarify"})
        ids = getattr(q, "ambiguous_ids", [])
        q.doc_ids = list(ids)
        q.needs_clarification = False
        clarify_options = q.clarify_options
        names = "; ".join(KB.documents[i].title for i in ids)
        ambiguity_hint = (f"Из вопроса нельзя однозначно определить договор: подходят {names}. Ответь по КАЖДОМУ "
                          f"из них отдельно (сначала назови контрагента и номер договора, потом условие со ссылкой) "
                          f"и в конце предложи уточнить, о каком договоре идёт речь.")

    hits = _search(q)
    context, blocks = build_context(q, hits)
    debug["hits"] = [h.chunk.chunk_id for h in hits]

    hints = [ambiguity_hint] if ambiguity_hint else []
    if q.doc_ids and not ambiguity_hint:
        hints.append("Документы, о которых, вероятно, идёт речь: " +
                     "; ".join(KB.documents[d].title for d in q.doc_ids) + ".")
    if q.event_date:
        hints.append(f"В вопросе указан момент события: «{q.event_date}» — отвечай по редакциям, "
                     f"действовавшим на эту дату.")
        hints += edition_hints(q, blocks, top_docs={h.chunk.doc_id for h in hits[:4]},
                               top_chunks={h.chunk.chunk_id for h in hits[:4]})
    focus = set(q.doc_ids) | {h.chunk.doc_id for h in hits[:6]}
    for did in list(focus):
        focus.update(KB.documents[did].related)
    if not q.event_date:
        hints += current_edition_hints(blocks, focus)
    hints += threshold_hints(q.last_user_message, blocks, focus)
    hints += premise_hints(q.last_user_message, context)
    hints += missing_reference_hints(blocks, top_docs={h.chunk.doc_id for h in hits[:3]})
    user = prompts.USER_TEMPLATE.format(
        today=dt.date.today().strftime("%d.%m.%Y"),
        hints=("\n".join(hints) + "\n") if hints else "",
        context=context or "(по запросу ничего не найдено)",
        dialog=_dialog_text(q.history), question=q.last_user_message)
    msgs = [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": user}]

    raw = llm_client.chat(msgs)
    if is_canned(raw):
        # заготовка/пустой ответ вместо ответа по документам — один повтор с уточнением задачи
        debug["retry"] = "canned"
        try:
            raw2 = llm_client.chat(msgs + [{"role": "assistant", "content": raw or "…"},
                                           {"role": "user", "content": prompts.RETRY_TEMPLATE}], temperature=0.3)
            if raw2 and not is_canned(raw2):
                raw = raw2
        except Exception as exc:
            log(f"[core] повтор после заготовки не удался: {exc}")
    data = _parse_json(raw)
    text = _clean_text(str(data.get("answer", "")))

    problems = check_answer(text, context + "\n" + q.last_user_message) if text else []
    if problems and not fast:
        debug["fix"] = problems
        fix_msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": prompts.FIX_TEMPLATE.format(problems=", ".join(problems))}]
        try:
            raw_fix = llm_client.chat(fix_msgs)
            data2 = _parse_json(raw_fix)
            text2 = _clean_text(str(data2.get("answer", "")))
            # исправление не должно превращать ответ в отказ
            if text2 and not is_canned(raw_fix) and not (is_refusal(text2, data2) and not is_refusal(text, data)):
                data, text, raw = data2, text2, raw_fix
            else:
                trimmed = drop_sentences(text, problems)   # убираем предложения с неподтверждёнными числами
                if trimmed:
                    text = trimmed
                    debug["trimmed"] = True
                elif text2 and not is_canned(raw_fix):
                    data, text, raw = data2, text2, raw_fix   # честный отказ лучше выдумки
        except Exception as exc:
            log(f"[core] повторная генерация не удалась: {exc}")
            trimmed = drop_sentences(text, problems)
            if trimmed:
                text = trimmed

    # ложный отказ: модель говорит «нет сведений», хотя найдены документы по теме — просим перепроверить
    if not fast and blocks and (is_refusal(text, data) or not text) and _relevant_found(q, hits):
        debug["reconsider"] = True
        frag = "\n".join("- " + KB.documents[h.chunk.doc_id].title + ", " + h.chunk.section + ": " +
                         " ".join(h.chunk.text.split())[:350] for h in hits[:4])
        rc_msgs = msgs + [{"role": "assistant", "content": raw or "{}"},
                          {"role": "user", "content": prompts.RECONSIDER_TEMPLATE.format(fragments=frag)}]
        try:
            raw_rc = llm_client.chat(rc_msgs)
            data3 = _parse_json(raw_rc)
            text3 = _clean_text(str(data3.get("answer", "")))
            if text3 and not is_canned(raw_rc):
                bad3 = check_answer(text3, context + "\n" + q.last_user_message)
                if not is_refusal(text3, data3) and not bad3:
                    data, text = data3, text3
                    debug["reconsider"] = "answered"
                elif not text:
                    data, text = data3, text3
        except Exception as exc:
            log(f"[core] перепроверка отказа не удалась: {exc}")

    # самопроверка по чек-листу: второй взгляд модели на свой ответ
    if not fast and text and not is_refusal(text, data) and blocks:
        try:
            cr_msgs = msgs + [{"role": "assistant", "content": json.dumps(
                                  {"status": data.get("status", "answer"), "answer": text, "used": data.get("used", [])},
                                  ensure_ascii=False)},
                              {"role": "user", "content": prompts.CRITIC_TEMPLATE}]
            raw_cr = llm_client.chat(cr_msgs)
            data4 = _parse_json(raw_cr)
            text4 = _clean_text(str(data4.get("answer", "")))
            ok = (text4 and not is_canned(raw_cr) and not is_refusal(text4, data4)
                  and not check_answer(text4, context + "\n" + q.last_user_message))
            if ok and text4.strip() != text.strip():
                data, text = data4, text4
                debug["critic"] = "changed"
            elif not ok:
                debug["critic"] = "rejected"
        except Exception as exc:
            log(f"[core] самопроверка не удалась: {exc}")

    # отказ без опоры на документы: ни договор не назван, ни найденное не относится к вопросу
    if is_refusal(text, data) and not _relevant_found(q, hits) and len(text) < 220:
        text = ("В базе знаний нет документов по этому вопросу, поэтому ответить на него по документам я не могу. "
                "Я отвечаю на вопросы о договорах компании, внутренних регламентах и нормах законодательства из базы "
                "знаний; по этому вопросу лучше обратиться к профильному специалисту.")
        data["status"] = "refusal"

    text, clause_fixes = fix_clause_numbers(text, blocks)
    if clause_fixes:
        debug["clause_fix"] = clause_fixes
    status = data.get("status") if data.get("status") in ("answer", "partial", "refusal", "clarify") else "answer"
    if not text:
        text, status = config.FALLBACK_TEXT, "refusal"
    used = data.get("used") if isinstance(data.get("used"), list) else []
    sources = [] if status in ("refusal", "clarify") and not used else _sources(text, blocks, used)
    return Answer(status=status, text=text, sources=sources, clarify_options=clarify_options, debug=debug)
