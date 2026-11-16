"""Структуры данных ядра (контракты между модулями, бэкендом и сабмитом)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str            # "user" | "assistant"
    content: str


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>#<пункт>"
    doc_id: str
    doc_title: str       # "Договор поставки № П-17/2024 с ООО «Альфа-Трейд»"
    doc_type: str        # contract | amendment | regulation | memo | letter | law
    doc_status: str      # active | expired
    section: str         # "п. 6.1", "ст. 70", "ст. 164, п. 2"
    text: str            # текст фрагмента дословно из базы
    order: int           # порядковый номер фрагмента в документе


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    status: str
    text: str                          # полный текст (очищенный)
    short_ref: str = ""                # "договор № П-17/2024", "ТК РФ, ст. 70"
    number: Optional[str] = None       # номер договора
    counterparty: Optional[str] = None
    date: Optional[str] = None         # дата документа ГГГГ-ММ-ДД
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    amends: Optional[str] = None       # doc_id договора, который меняет доп. соглашение
    related: list = field(default_factory=list)  # doc_id связанных документов
    chunk_ids: list = field(default_factory=list)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class ParsedQuery:
    last_user_message: str
    history: list                       # list[Message] без последней реплики
    search_query: str
    doc_ids: list = field(default_factory=list)
    event_date: Optional[str] = None
    needs_clarification: bool = False
    clarify_question: Optional[str] = None
    clarify_options: list = field(default_factory=list)


@dataclass
class Source:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    quote: str
    doc_status: str = "active"


@dataclass
class Answer:
    status: str                         # answer | partial | refusal | clarify
    text: str
    sources: list = field(default_factory=list)          # list[Source]
    clarify_options: list = field(default_factory=list)  # [{"label", "message"}]
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
