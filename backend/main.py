"""Бэкенд веб-чата. Запуск из папки backend/:  uvicorn main:app --reload --port 8000

Ядро (core/) лежит в корне проекта, на уровень выше backend/.

CORE_MODE=mock  — никакого импорта core/llm_client и никаких соединений с GigaChat:
                  все эндпоинты обслуживаются заглушками внутри этого файла.
CORE_MODE=real  — импортируется legal_core, /api/chat вызывает реальную модель,
                  /api/documents читает реальную базу.

Переменные окружения (GIGACHAT_AUTH_KEY, SSL_VERIFY, CORE_MODE, DATA_DIR,
FRONTEND_ORIGIN) задаются в оболочке перед запуском uvicorn, например:
    export GIGACHAT_AUTH_KEY="..."
    export SSL_VERIFY="0"
    export CORE_MODE="real"
    uvicorn main:app --reload --port 8000
"""
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

logger = logging.getLogger("uvicorn.error")

CORE_MODE = os.getenv("CORE_MODE", "real").strip().lower()
IS_MOCK = (CORE_MODE == "mock")

logger.info("CORE_MODE=%s", CORE_MODE)

if not IS_MOCK:
    sys.path.insert(0, str(PROJECT_ROOT / "core"))
    import legal_core  # noqa: E402
    from llm_client import LLMUnavailable  # noqa: E402


app = FastAPI(title="Юридический ассистент")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


@app.on_event("startup")
def startup() -> None:
    if IS_MOCK:
        logger.info("CORE_MODE=mock: init_core пропущен, соединений с GigaChat нет")
        return
    data_dir = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
    legal_core.init_core(data_dir)


@app.get("/api/health")
def health():
    return {"status": "ok", "core_mode": CORE_MODE}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.messages[-1].role != "user":
        raise HTTPException(422, "Последняя реплика должна быть от пользователя")

    if IS_MOCK:
        last = req.messages[-1].content
        return {
            "answer": f"[MOCK] Ответ на вопрос: {last}",
            "sources": [],
            "clarification": None,
        }

    try:
        answer = legal_core.answer_question(
            [m.model_dump() for m in req.messages], allow_clarify=True
        )
    except LLMUnavailable as exc:
        raise HTTPException(
            503, {"code": "llm_unavailable", "message": str(exc)[:300]}
        )
    except Exception as exc:
        raise HTTPException(
            500, {"code": "core_error", "message": f"{type(exc).__name__}: {exc}"[:300]}
        )
    data = answer.to_dict()
    data.pop("debug", None)
    return data


@app.get("/api/documents")
def documents():
    if IS_MOCK:
        return [
            {"id": "mock-doc-1", "title": "[MOCK] Договор аренды"},
            {"id": "mock-doc-2", "title": "[MOCK] Трудовой договор"},
        ]
    return legal_core.list_documents()


@app.get("/api/documents/{doc_id}")
def document(doc_id: str):
    if IS_MOCK:
        return {
            "id": doc_id,
            "title": f"[MOCK] Документ {doc_id}",
            "items": [
                {"number": "1", "text": "[MOCK] Пункт 1"},
                {"number": "2", "text": "[MOCK] Пункт 2"},
            ],
        }
    try:
        return legal_core.get_document(doc_id)
    except KeyError:
        raise HTTPException(
            404, {"code": "not_found", "message": f"Нет документа {doc_id}"}
        )