"""Чтение базы знаний: документы -> фрагменты (пункты договоров, части статей) с метаданными."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import config
from llm_client import log
from schemas import Chunk, Document

KB_NAMES = ("knowledge-base.zip", "kb.zip", "knowledge_base.zip")

MONTHS = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7,
          "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}

# Строки, обращённые к ИИ внутри документов, — это не содержание документа, а попытка
# управлять ассистентом. Такие строки вырезаются из базы.
INJECTION_RE = re.compile(
    r"(для\s+ии|ии-ассистент|ai[- ]assistant|игнорируй|ignore\s+(all|previous)|"
    r"не\s+упоминай\s+эт\w*\s+инструкц|системн\w+\s+(сообщени|инструкц))", re.I)

LAW_NAMES = {
    "ТК": "ТК РФ", "НК": "НК РФ", "КоАП": "КоАП РФ", "ГК": "ГК РФ",
    "152ФЗ": "Федеральный закон № 152-ФЗ «О персональных данных»",
    "МРОТ": "Федеральный закон № 82-ФЗ «О минимальном размере оплаты труда»",
}

CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S")            # 3.1. / 3.1 текст
SECTION_RE = re.compile(r"^(\d+)\.\s+([А-ЯЁA-Z][^.]{0,80})$")   # 3. Арендная плата
ARTICLE_RE = re.compile(r"^Статья\s+\d+(?:\.\d+)*\.\s+\S")   # «Статья 70. Испытание…»
LAW_PART_RE = re.compile(r"^(\d+(?:\.\d+)?)\.\s+\S")            # 1. / 1.1. часть/пункт статьи
EDITORIAL_RE = re.compile(r"^(Статья|Часть|Пункт|Подпункт|Абзац|Положения|Федеральным|"
                          r"О конституционно|См\.|Информация об изменениях)\b.*"
                          r"(изменен|утратил|дополнен|применяются|редакци|смысл|вступающ|признан)", re.I)


# ---------------------------------------------------------------- чтение файлов
def find_kb_zip(data_dir: Path) -> Path:
    roots = [data_dir, Path(__file__).resolve().parent / data_dir, Path.cwd()]
    for root in roots:
        for name in KB_NAMES:
            if (root / name).is_file():
                return root / name
    # любой zip в каталоге данных
    for root in roots:
        if root.is_dir():
            for p in sorted(root.glob("*.zip")):
                return p
    raise FileNotFoundError(f"Не найден архив базы знаний в {data_dir}")


def _read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(t.text or "" for t in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def read_archive(path: Path) -> list:
    files = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            if name.endswith("/") or "__MACOSX" in name:
                continue
            # имена в zip бывают в cp866 без флага UTF-8
            if not (info.flag_bits & 0x800):
                try:
                    name = name.encode("cp437").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    try:
                        name = name.encode("cp437").decode("cp866")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
            suffix = Path(name).suffix.lower()
            if suffix not in {".txt", ".md", ".docx"}:
                continue
            raw = archive.read(info)
            try:
                if suffix == ".docx":
                    text = _read_docx(raw)
                else:
                    try:
                        text = raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text = raw.decode("cp1251")
            except Exception as exc:
                log(f"[kb] пропущен {name}: {exc}")
                continue
            if text.strip():
                files.append((Path(name).name, text.replace("\r\n", "\n").replace("\r", "\n")))
    files.sort()
    return files


# ---------------------------------------------------------------- утилиты
def slug(text: str) -> str:
    table = str.maketrans({"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
                           "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
                           "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
                           "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
                           "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya"})
    s = text.lower().translate(table)
    s = re.sub(r"[^a-z0-9.]+", "-", s).strip("-")
    return s[:80] or "doc"


def norm_number(s: str) -> str:
    """Номер договора в сравнимом виде: «П-17/2024», «п 17-2024» -> «п172024»."""
    s = s.lower().replace("ё", "е").translate(str.maketrans("aekmhopctyx", "аекмнорстух"))
    return re.sub(r"[^0-9а-я]", "", s)


def parse_ru_date(text: str):
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.search(r"\b(\d{1,2})\s+([а-яё]+)\s+(\d{4})\s*г", text.lower())
    if m:
        for stem, num in MONTHS.items():
            if m.group(2).startswith(stem) and (stem != "ма" or m.group(2).startswith("ма") and len(m.group(2)) <= 4):
                return f"{int(m.group(3)):04d}-{num:02d}-{int(m.group(1)):02d}"
    return None


def clean_lines(text: str, doc_name: str) -> list:
    out = []
    for line in text.split("\n"):
        line = line.strip().replace(" ", " ")
        if not line:
            continue
        if INJECTION_RE.search(line):
            log(f"[kb] вырезана строка-инструкция для ИИ в «{doc_name}»")
            continue
        out.append(line)
    return out


# ---------------------------------------------------------------- документы компании
def _company_meta(fname: str, lines: list) -> dict:
    head = " ".join(lines[:6])
    low = fname.lower()
    meta = {"status": "expired" if ("утратил" in low or "утратила силу" in head.lower()[:200]) else "active"}
    m = re.search(r"№\s*([А-ЯA-ZЁ]{1,4}-\d+/\d{4})", head)
    number = m.group(1) if m else None
    cp = None
    for cand in re.findall(r"ООО\s+«([^»]+)»", " ".join(lines[:8])):
        if cand.lower() not in ("меридиан",):
            cp = f"ООО «{cand}»"
            break
    dated = [l for l in lines[:5] if " — " in l and re.search(r"\d{4}\s*г", l)]
    meta["date"] = parse_ru_date(dated[0] if dated else " ".join(lines[1:4]))

    if "доп" in low and "соглаш" in low:
        meta["doc_type"] = "amendment"
        m2 = re.search(r"к договору[^№]*№\s*([А-ЯA-ZЁ]{1,4}-\d+/\d{4})", head)
        meta["amends_number"] = m2.group(1) if m2 else None
        meta["number"] = None
        meta["title"] = (lines[0].title() if lines[0].isupper() else lines[0]) + (" " + lines[1] if len(lines) > 1 else "")
        meta["counterparty"] = cp
        meta["short_ref"] = "доп. соглашение № 1 к договору № " + (meta["amends_number"] or "")
        meta["id_hint"] = "dop-" + (meta["amends_number"] or Path(fname).stem)
        if lines[0].upper().startswith("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ"):
            meta["short_ref"] = lines[0].capitalize().replace("Дополнительное", "доп.").replace("соглашение", "соглашение") + \
                (" к договору № " + meta["amends_number"] if meta.get("amends_number") else "")
        return meta
    if "договор" in low and number:
        kind = lines[0].strip().capitalize()
        kind = re.sub(r"\s*№.*$", "", kind)
        meta.update(doc_type="contract", number=number, counterparty=cp,
                    title=f"{kind} № {number}" + (f" с {cp}" if cp else ""),
                    short_ref=f"договор № {number}")
        return meta
    if "регламент" in low:
        ed = re.search(r"[Рр]едакция от (\d{2}\.\d{2}\.\d{4})", " ".join(lines[:5]))
        ed_date = parse_ru_date(ed.group(1)) if ed else None
        title = "Регламент согласования договоров ООО «Меридиан»" if "согласован" in low else lines[0].capitalize()
        title += f" (ред. от {ed.group(1)})" if ed else ""
        meta["id_hint"] = "reglament-" + (ed_date[:4] if ed_date else Path(fname).stem)
        meta.update(doc_type="regulation", title=title, valid_from=ed_date,
                    short_ref=f"Регламент (ред. от {ed.group(1)})" if ed else "Регламент")
        lost = re.search(r"УТРАТИЛА СИЛУ с (\d{2}\.\d{2}\.\d{4})", " ".join(lines[:2]), re.I)
        if lost:
            meta["valid_to"] = parse_ru_date(lost.group(1))
        return meta
    if "письмо" in low:
        m3 = re.search(r"Исх\.\s*№\s*(\S+)\s+от\s+(\d{2}\.\d{2}\.\d{4})", head)
        title = "Письмо службы комплаенс" + (f" № {m3.group(1)} от {m3.group(2)}" if m3 else "")
        meta["id_hint"] = "pismo-" + (m3.group(1) if m3 else "1")
        meta.update(doc_type="letter", title=title, short_ref=title,
                    date=parse_ru_date(m3.group(2)) if m3 else meta["date"])
        return meta
    if "записк" in low:
        m4 = re.search(r"СЛУЖЕБНАЯ ЗАПИСКА\s*№\s*(\S+)\s+от\s+(\d{2}\.\d{2}\.\d{4})", head, re.I)
        title = "Служебная записка юридического отдела" + (f" № {m4.group(1)} от {m4.group(2)}" if m4 else "")
        meta["id_hint"] = "zapiska-" + (m4.group(1) if m4 else "1")
        meta.update(doc_type="memo", title=title, short_ref=title,
                    date=parse_ru_date(m4.group(2)) if m4 else meta["date"])
        return meta
    stem = re.sub(r"^.*?__", "", Path(fname).stem)
    meta.update(doc_type="memo", title=stem, short_ref=stem)
    return meta


def _split_company(doc_id: str, title: str, lines: list):
    """Пункты вида 3.1. — отдельные фрагменты; шапка — преамбула; без нумерации — весь документ."""
    chunks = []
    has_clauses = sum(1 for l in lines if CLAUSE_RE.match(l)) >= 3
    if not has_clauses:
        return [("весь документ", "", "\n".join(lines))]
    preamble, section_title = [], ""
    current = None
    for line in lines:
        sm = SECTION_RE.match(line)
        cm = CLAUSE_RE.match(line)
        if cm:
            if current:
                chunks.append(current)
            current = [f"п. {cm.group(1)}", section_title, line]
        elif sm and not cm:
            if current:
                chunks.append(current)
                current = None
            section_title = line
            # строка раздела без пунктов (например «7. Реквизиты…») — копим как отдельный кусок
            current = [f"разд. {sm.group(1)}", line, line]
        elif current is None:
            preamble.append(line)
        else:
            current[2] += "\n" + line
    if current:
        chunks.append(current)
    out = []
    if preamble:
        out.append(("преамбула", "", "\n".join(preamble)))
    for sec, stitle, text in chunks:
        if sec.startswith("разд.") and text.strip() == stitle.strip():
            continue  # пустой заголовок раздела
        out.append((sec, stitle, text))
    return out


# ---------------------------------------------------------------- законы
def law_topic(code: str, art: str) -> str:
    """Тема статьи по номеру (главы кодексов), чтобы поиск находил её по привычным словам и аббревиатурам."""
    try:
        n = float(art.split(".")[0])
    except ValueError:
        return ""
    if code == "НК":
        if 106 <= n <= 142:
            return "налоговые правонарушения, ответственность, налоговые санкции, штрафы"
        if 143 <= n <= 177:
            return "НДС, налог на добавленную стоимость"
        if 207 <= n <= 233:
            return "НДФЛ, налог на доходы физических лиц"
        if 246 <= n <= 333:
            return "налог на прибыль организаций"
    if code == "ТК":
        if 56 <= n <= 84:
            return "трудовой договор, приём на работу"
        if 91 <= n <= 105:
            return "рабочее время, сверхурочная работа"
        if 129 <= n <= 158:
            return "оплата труда, заработная плата"
    if code == "КоАП" and 13 <= n < 14:
        return "административная ответственность, штрафы, персональные данные"
    if code == "МРОТ":
        return "МРОТ, минимальный размер оплаты труда, минимальная зарплата"
    if code == "152ФЗ":
        return "персональные данные, ПДн, согласие на обработку"
    return ""


def _law_meta(fname: str, lines: list) -> dict:
    stem = re.sub(r"^.*?__", "", Path(fname).stem)
    parts = stem.split("_")
    code = parts[0]
    art = parts[1].replace("ст", "") if len(parts) > 1 else ""
    law_name = LAW_NAMES.get(code, code)
    art_title = ""
    for line in lines[:25]:
        if ARTICLE_RE.match(line) and not EDITORIAL_RE.match(line):
            art_title = line
            break
    short = f"{law_name}, ст. {art}" if art else law_name
    if len(art_title) > 140:
        art_title = art_title[:140].rsplit(" ", 1)[0] + "…"
    title = f"{law_name}. {art_title}" if art_title else short
    return {"doc_type": "law", "status": "active", "title": title, "short_ref": short, "article": art,
            "id_hint": f"{code}-st{art}", "topic": law_topic(code, art)}


def _split_law(short_ref: str, article: str, lines: list):
    body, started = [], False
    for line in lines:
        if line.startswith(("Источник:", "URL:", "Дата выгрузки:")):
            continue
        if not started and ARTICLE_RE.match(line) and not EDITORIAL_RE.match(line):
            started = True  # строку статьи оставляем: в ней бывает сама норма (МРОТ ст. 1)
        body.append(line)
    if not body:
        body = lines

    numbered = sum(1 for l in body if LAW_PART_RE.match(l)) >= 2
    parts = []          # [(label, [lines])]
    notes = []          # редакционные пометки копим и приклеиваем к следующей части
    current = None
    for line in body:
        if EDITORIAL_RE.match(line) or line.startswith("Информация об изменениях"):
            notes.append(f"[Примечание: {line}]")
            continue
        m = LAW_PART_RE.match(line) if numbered else None
        new_part = bool(m) or (not numbered and not re.match(r"^[а-яё(]", line) and current is not None
                               and not current[1][-1].rstrip().endswith((":", ";")))
        if current is None or new_part:
            if current:
                parts.append(current)
            label = f"п. {m.group(1)}" if m else ""
            current = [label, notes + [line]]
            notes = []
        else:
            current[1].extend(notes + [line])
            notes = []
    if current:
        parts.append(current)
    if notes and parts:
        parts[-1][1].extend(notes)

    # упаковка частей во фрагменты до CHUNK_MAX_CHARS
    out = []
    limit = config.CHUNK_MAX_CHARS
    base = f"ст. {article}" if article else short_ref
    buf_lines, buf_labels = [], []

    def flush():
        if buf_lines:
            labels = [l for l in buf_labels if l]
            sec = base + (f", {labels[0]}" if len(labels) == 1 else f", {labels[0]}–{labels[-1].replace('п. ', '')}" if labels else "")
            out.append((sec, "", "\n".join(buf_lines)))
        buf_lines.clear()
        buf_labels.clear()

    for label, plines in parts:
        text = "\n".join(plines)
        if len(text) > limit:
            flush()
            piece, size = [], 0
            for pl in plines:
                if size + len(pl) > limit and piece:
                    buf_lines.extend(piece); buf_labels.append(label); flush()
                    piece, size = [f"(продолжение {label or base})"], 0
                piece.append(pl); size += len(pl) + 1
            buf_lines.extend(piece); buf_labels.append(label); flush()
            continue
        if sum(len(x) + 1 for x in buf_lines) + len(text) > limit:
            flush()
        buf_lines.extend(plines)
        buf_labels.append(label)
    flush()
    return out


# ---------------------------------------------------------------- база целиком
class KnowledgeBase:
    def __init__(self, documents: dict, chunks: list):
        self.documents = documents                  # doc_id -> Document
        self.chunks = chunks                        # list[Chunk]
        self.by_id = {c.chunk_id: c for c in chunks}
        self.contracts = [d for d in documents.values() if d.doc_type == "contract"]

    def chunks_of(self, doc_id: str) -> list:
        return [self.by_id[cid] for cid in self.documents[doc_id].chunk_ids]


def load_kb(data_dir: Path = None) -> KnowledgeBase:
    path = find_kb_zip(Path(data_dir or config.DATA_DIR))
    files = read_archive(path)
    documents, chunks = {}, []
    for fname, raw in files:
        lines = clean_lines(raw, fname)
        if not lines:
            continue
        is_law = fname.lower().startswith("закон") or bool(re.match(r"^Источник:", lines[0]))
        if is_law:
            meta = _law_meta(fname, lines)
            pieces = _split_law(meta["short_ref"], meta["article"], lines)
        else:
            meta = _company_meta(fname, lines)
            pieces = None
        doc_id = slug(meta.get("number") or meta.get("id_hint") or meta.get("short_ref") or fname).replace(".", "-")
        while doc_id in documents:
            doc_id += "-2"
        doc = Document(doc_id=doc_id, title=meta["title"], doc_type=meta["doc_type"],
                       status=meta.get("status", "active"), text="\n".join(lines),
                       short_ref=meta.get("short_ref", ""), number=meta.get("number"),
                       counterparty=meta.get("counterparty"), date=meta.get("date"),
                       valid_from=meta.get("valid_from"), valid_to=meta.get("valid_to"))
        doc._amends_number = meta.get("amends_number")
        doc.topic = meta.get("topic", "")
        if pieces is None:
            pieces = _split_company(doc_id, doc.title, lines)
        for i, (sec, stitle, text) in enumerate(pieces):
            key = (sec.lower().replace("ст. ", "st").replace(", п. ", "-p").replace("п. ", "")
                   .replace("разд. ", "r").replace("–", "_"))
            key = slug(key) if re.search(r"[а-я]", key) else re.sub(r"[^0-9a-z._-]+", "", key)
            cid = f"{doc_id}#{key or i}"
            while cid in {c.chunk_id for c in chunks}:
                cid += f"-{i}"
            ch = Chunk(chunk_id=cid, doc_id=doc_id, doc_title=doc.title, doc_type=doc.doc_type,
                       doc_status=doc.status, section=sec,
                       text=(stitle + "\n" + text) if stitle and stitle not in text else text, order=i)
            chunks.append(ch)
            doc.chunk_ids.append(cid)
        documents[doc_id] = doc

    # связи: доп. соглашение -> договор; редакции одного регламента
    by_number = {norm_number(d.number): d for d in documents.values() if d.number}
    for d in documents.values():
        num = getattr(d, "_amends_number", None)
        if d.doc_type == "amendment" and num and norm_number(num) in by_number:
            target = by_number[norm_number(num)]
            d.amends = target.doc_id
            d.counterparty = d.counterparty or target.counterparty
            d.title = f"Дополнительное соглашение № 1 к договору № {target.number} с {target.counterparty}" \
                if "№ 1" in d.text[:60] else d.title
            target.related.append(d.doc_id)
            d.related.append(target.doc_id)
    regs = [d for d in documents.values() if d.doc_type == "regulation"]
    for a in regs:
        for b in regs:
            if a is not b and a.title.split(" (ред")[0] == b.title.split(" (ред")[0]:
                a.related.append(b.doc_id)
    # у утратившей силу редакции valid_to; у действующей — valid_from нового = valid_to старого
    for a in regs:
        if a.status == "active" and not a.valid_from:
            m = re.search(r"Действует с (\d{2}\.\d{2}\.\d{4})", a.text[:400])
            a.valid_from = parse_ru_date(m.group(1)) if m else None
    # обновим подписи фрагментов после переименований
    for c in chunks:
        c.doc_title = documents[c.doc_id].title
    log(f"[kb] {len(documents)} документов, {len(chunks)} фрагментов из {path.name}")
    return KnowledgeBase(documents, chunks)
