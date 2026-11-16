"""Запросы к API организаторов: генерация, эмбеддинги, реранкер. С повторами при ошибках."""
from __future__ import annotations

import random
import sys
import threading
import time

import config

_client = None
_lock = threading.Lock()

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMUnavailable(RuntimeError):
    pass


def _get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import httpx  # импорт здесь, чтобы модули грузились и без httpx (локальные тесты)
                verify = config.SSL_CA_BUNDLE or config.SSL_VERIFY
                _client = httpx.Client(
                    verify=verify,
                    timeout=httpx.Timeout(90.0, connect=15.0),
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                )
    return _client


_token = {"value": "", "expires": 0.0}
_token_lock = threading.Lock()


def gigachat_token(force: bool = False) -> str:
    """Токен GigaChat по ключу авторизации; живёт 30 минут — обновляем заранее, потокобезопасно."""
    import uuid
    with _token_lock:
        if not force and _token["value"] and time.time() < _token["expires"] - 120:
            return _token["value"]
        r = _get_client().post(config.GIGACHAT_AUTH_URL, data={"scope": config.GIGACHAT_SCOPE}, timeout=30.0,
                               headers={"Authorization": "Basic " + config.GIGACHAT_AUTH_KEY,
                                        "RqUID": str(uuid.uuid4()), "Accept": "application/json",
                                        "Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code != 200:
            raise LLMUnavailable(f"GigaChat: не удалось получить токен, HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        _token["value"] = data["access_token"]
        exp = data.get("expires_at")              # в миллисекундах
        _token["expires"] = exp / 1000 if exp else time.time() + 25 * 60
        log("[llm] получен токен GigaChat")
        return _token["value"]


def _post(path: str, key: str, payload: dict, attempts: int = 3, timeout: float = 90.0) -> dict:
    if config.GIGACHAT:
        key = gigachat_token()
    if not config.API_BASE or not key:
        raise LLMUnavailable("Не заданы LLM_BASE_URL / ключ API")
    client = _get_client()
    last_error = None
    for attempt in range(attempts):
        try:
            response = client.post(config.API_BASE + path,
                                   headers={"Authorization": "Bearer " + key},
                                   json=payload, timeout=timeout)
            if response.status_code == 401 and config.GIGACHAT and attempt < attempts - 1:
                key = gigachat_token(force=True)      # токен истёк — обновляем и повторяем
                continue
            if response.status_code in RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            elif response.status_code >= 400:
                raise LLMUnavailable(f"HTTP {response.status_code} {path}: {response.text[:300]}")
            else:
                return response.json()
        except LLMUnavailable:
            raise
        except Exception as exc:  # сеть, тайм-аут, кривой JSON
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            time.sleep(1.5 * (2 ** attempt) + random.random())
    raise LLMUnavailable(f"{path}: {last_error}")


def chat(messages: list, max_tokens: int | None = None, temperature: float | None = None) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or config.MAX_TOKENS,
    }
    data = _post("/chat/completions", config.LLM_KEY, payload, timeout=150.0)
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        raise LLMUnavailable(f"Неожиданный ответ chat: {str(data)[:300]}")
    # ответ оборвался на лимите (часто из-за долгих рассуждений модели) — один повтор с большим лимитом
    if (finish == "length" or not content.strip()) and payload["max_tokens"] < 8000:
        log(f"[llm] ответ оборван/пуст (finish={finish}), повтор с большим лимитом")
        payload["max_tokens"] = min(8000, payload["max_tokens"] * 2)
        data2 = _post("/chat/completions", config.LLM_KEY, payload, timeout=180.0)
        try:
            content2 = data2["choices"][0]["message"].get("content") or ""
            if content2.strip():
                content = content2
        except (KeyError, IndexError, TypeError):
            pass
    return strip_thinking(content).strip()


def strip_thinking(text: str) -> str:
    """Убирает рассуждения модели, если они пришли внутри текста ответа (<think>…</think>)."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    if text.lstrip().lower().startswith("<think>"):
        return ""
    return text


def embed(texts: list, batch_size: int = 32) -> list:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:config.EMB_MAX_CHARS] for t in texts[start:start + batch_size]]
        payload = {"model": config.EMB_MODEL, "input": batch}
        if config.EMB_ENCODING_FORMAT:
            payload["encoding_format"] = config.EMB_ENCODING_FORMAT
        data = _post("/embeddings", config.EMBED_KEY, payload, timeout=60.0)
        items = data.get("data") if isinstance(data, dict) else None
        if not items or len(items) != len(batch):
            raise LLMUnavailable(f"Неожиданный ответ embeddings: {str(data)[:300]}")
        items = sorted(items, key=lambda x: x.get("index", 0))
        vectors.extend(item["embedding"] for item in items)
    return vectors


def rerank(query: str, documents: list, top_n: int) -> list:
    """Возвращает [(индекс документа, оценка)] по убыванию оценки."""
    data = _post("/rerank", config.RERANK_KEY,
                 {"model": config.RERANKER_MODEL, "query": query,
                  "documents": [d[:4000] for d in documents], "top_n": min(top_n, len(documents))},
                 timeout=60.0)
    items = data.get("results") or data.get("data") or []
    out = []
    for item in items:
        idx = item.get("index")
        score = item.get("relevance_score", item.get("score", 0.0))
        if isinstance(idx, int) and 0 <= idx < len(documents):
            out.append((idx, float(score or 0.0)))
    if not out:
        raise LLMUnavailable(f"Неожиданный ответ rerank: {str(data)[:300]}")
    out.sort(key=lambda x: -x[1])
    return out


def log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)
