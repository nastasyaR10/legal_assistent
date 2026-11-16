"""Поиск: BM25 (по словам) + векторы bge-m3 (по смыслу) -> слияние RRF -> реранкер."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

import numpy as np

import config
import llm_client
from llm_client import log
from schemas import ParsedQuery, SearchHit

WORD_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[a-zа-яё]+", re.I)
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам
ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже
себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая
много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
между ли нужно ли какие какая какой каков сколько который которые это""".split())


def stem(word: str) -> str:
    w = word.lower().replace("ё", "е")
    if w.isdigit() or len(w) <= 4:
        return w
    return w[:6] if len(w) > 7 else w[:max(4, len(w) - 2)]


def tokenize(text: str) -> list:
    return [stem(w) for w in WORD_RE.findall(text) if w.lower() not in STOP]


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list) -> list:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def chunk_search_text(chunk, doc) -> str:
    """Текст фрагмента для индекса: с названием документа и пунктом, чтобы фрагмент был понятен сам по себе."""
    extra = f" {doc.counterparty}" if doc.counterparty else ""
    topic = getattr(doc, "topic", "")
    topic = f" ({topic})" if topic else ""
    tags = ""
    if re.search(r"в случаях, не указанных|если иное не (?:установлено|предусмотрено)", chunk.text):
        tags = " [общее правило, основная (общая) ставка или норма]"
    return f"{doc.title}{topic}{extra}. {chunk.section}{tags}.\n{chunk.text}"


class Retriever:
    def __init__(self, kb):
        self.kb = kb
        self.texts = [chunk_search_text(c, kb.documents[c.doc_id]) for c in kb.chunks]
        self.bm25 = BM25([tokenize(t) for t in self.texts])
        self.matrix = None
        self._build_vectors()

    # ------------------------------------------------------------ векторы
    def _build_vectors(self) -> None:
        key = hashlib.md5((config.EMB_MODEL + "\n".join(self.texts)).encode("utf-8")).hexdigest()
        cache = config.INDEX_CACHE_DIR / f"emb_{key}.npy"
        try:
            if cache.is_file():
                self.matrix = np.load(cache)
                log(f"[retriever] векторы из кэша {cache}")
                return
        except Exception:
            pass
        try:
            vectors = llm_client.embed(self.texts)
            m = np.asarray(vectors, dtype=np.float32)
            m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
            self.matrix = m
            try:
                config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache, m)
            except Exception:
                pass
            log(f"[retriever] векторы посчитаны: {m.shape}")
        except Exception as exc:
            log(f"[retriever] эмбеддинги недоступны, только BM25: {exc}")
            self.matrix = None

    def _dense_scores(self, query: str):
        if self.matrix is None:
            return None
        try:
            q = np.asarray(llm_client.embed([query])[0], dtype=np.float32)
            q /= (np.linalg.norm(q) + 1e-9)
            return self.matrix @ q
        except Exception as exc:
            log(f"[retriever] эмбеддинг запроса не удался: {exc}")
            return None

    # ------------------------------------------------------------ поиск
    def search(self, query: ParsedQuery, top_n: int = None, exclude: set = None, use_rerank: bool = True) -> list:
        top_n = top_n or config.FINAL_TOP
        text = query.search_query
        n = len(self.kb.chunks)

        bm = self.bm25.scores(tokenize(text))
        bm_rank = sorted(range(n), key=lambda i: -bm[i])[:config.CANDIDATES]
        dense = self._dense_scores(text)
        dn_rank = list(np.argsort(-dense)[:config.CANDIDATES]) if dense is not None else []

        # слияние рангов (RRF)
        fused = Counter()
        for r, i in enumerate(bm_rank):
            if bm[i] > 0:
                fused[i] += 1.0 / (60 + r)
        for r, i in enumerate(dn_rank):
            fused[int(i)] += 1.0 / (60 + r)

        # документы, явно названные в вопросе: поднимаем их фрагменты
        target = set(query.doc_ids)
        for did in list(target):
            target.update(self.kb.documents[did].related)
        if target:
            for i, c in enumerate(self.kb.chunks):
                if c.doc_id in target:
                    fused[i] += 0.008 if c.doc_id in query.doc_ids else 0.004

        if exclude:
            for i, c in enumerate(self.kb.chunks):
                if c.doc_id in exclude:
                    fused.pop(i, None)
        pool = [i for i, _ in fused.most_common(config.RERANK_POOL)]
        if not pool:
            return []

        # реранкер
        order = None
        try:
            if not use_rerank or config.RERANKER_MODEL.lower() in ("none", "off", ""):
                raise RuntimeError("реранкер отключён")
            rr = llm_client.rerank(text, [self.texts[i] for i in pool], top_n=min(len(pool), 10))
            order = [(pool[idx], score) for idx, score in rr]
            ranked = {i for i, _ in order}   # остальные кандидаты — после реранкнутых, в порядке RRF
            order += [(i, -1.0 + fused[i]) for i in pool if i not in ranked]
            if target:  # мягко поднимаем фрагменты нужного договора и после реранкера
                order = [(i, s + (0.15 if self.kb.chunks[i].doc_id in target else 0.0)) for i, s in order]
                order.sort(key=lambda x: -x[1])
        except Exception as exc:
            log(f"[retriever] реранкер недоступен: {exc}")
        if order is None:
            order = [(i, fused[i]) for i in pool]

        hits = [SearchHit(chunk=self.kb.chunks[i], score=float(s)) for i, s in order[:top_n]]
        return hits
