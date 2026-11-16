"""Настройки: всё читается из переменных окружения платформы."""
import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", "data"))

# ---- Режим GigaChat (только для локальной разработки и сайта; на платформе не задаётся) ----
# Достаточно задать GIGACHAT_AUTH_KEY (ключ авторизации из личного кабинета) — токен ядро получит и обновит само.
GIGACHAT_AUTH_KEY = env("GIGACHAT_AUTH_KEY")
GIGACHAT = bool(GIGACHAT_AUTH_KEY)
GIGACHAT_SCOPE = env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_AUTH_URL = env("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")
            or ("https://gigachat.devices.sberbank.ru/api/v1" if GIGACHAT else "")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL",
                                                       "GigaChat-2" if GIGACHAT else "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "Embeddings" if GIGACHAT else "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "none" if GIGACHAT else "cohere/rerank-5-fast")   # у GigaChat нет реранкера

TEMPERATURE = env_float("TEMPERATURE", 0.1 if GIGACHAT else 0.0)
# не меньше 2500: ответ в JSON + запас, если модель «думает» перед ответом (рассуждения тоже тратят лимит)
MAX_TOKENS = max(2500, env_int("MAX_TOKENS", 2500, 128, 16384))
# организаторы: лимитов на параллельность нет; с самопроверкой на вопрос уходит 2–3 запроса — берём 8 потоков
MAX_WORKERS = max(8, env_int("MAX_WORKERS", 8, 1, 16))

# Поиск
CANDIDATES = 40          # сколько кандидатов берём из BM25 и из векторов
RERANK_POOL = 24         # сколько кандидатов отдаём реранкеру
FINAL_TOP = 8            # сколько фрагментов остаётся после реранкера
CHUNK_MAX_CHARS = 1500   # максимальный размер фрагмента закона
CONTEXT_MAX_CHARS = 75000   # широкий контекст: документы компании целиком + статьи законов

# Время: на платформе 20 минут на весь прогон
TIME_LIMIT_SEC = 20 * 60
SOFT_DEADLINE_SEC = 15.5 * 60   # после — без повторной проверки ответа
HARD_DEADLINE_SEC = 18.5 * 60   # после — сразу безопасный ответ

# Только для локальной разработки (на платформе не задаются и ни на что не влияют)
SSL_CA_BUNDLE = env("SSL_CA_BUNDLE")                 # путь к сертификату (например, НУЦ Минцифры для GigaChat)
SSL_VERIFY = env("SSL_VERIFY", "1") != "0"
EMB_ENCODING_FORMAT = env("EMB_ENCODING_FORMAT", "" if GIGACHAT else "float")  # пусто — не передавать параметр
EMB_MAX_CHARS = env_int("EMB_MAX_CHARS", 1200 if GIGACHAT else 6000, 200, 20000)  # обрезка текста перед эмбеддингом

INDEX_CACHE_DIR = Path(env("INDEX_CACHE_DIR", "/tmp/legal_index"))

FALLBACK_TEXT = ("Не удалось сформировать обоснованный ответ по базе знаний. "
                 "Рекомендуется обратиться к юристу.")
