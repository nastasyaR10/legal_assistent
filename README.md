# Юридический ИИ-ассистент ПСБ

**Что умеет:**
- Ищет положения в договорах, регламентах и законах из базы знаний
- Отвечает со ссылками на конкретные пункты и статьи
- Учитывает редакцию документа, действовавшую на дату события
- Показывает источники

## Запуск
Указать переменные LLM_BASE_URL, LLM_API_KEY

```bash
export CORE_MODE=real
docker compose up --build
```
URL: http://localhost:3000
Порт для бэкенда: localhost:8000
Порт для фронтенда: localhost:3000

## Архитектура проекта
```text
lawyer_assistent/
│
├── compose.yaml
├── requirements.txt
├── README.md
│
├── data/
│ └── knowledge-base.zip # база знаний
│
├── backend/ # FastAPI-обёртка над ядром
│ ├── init.py
│ ├── Dockerfile
│ └── main.py # /api/chat, /api/documents, /api/health
│
├── core/ # RAG-ядро
│ ├── legal_core.py # answer_question(): analyze → search → context → generate → verify
│ ├── retriever.py # BM25 + bge-m3 + RRF + реранкер
│ ├── kb_loader.py # knowledge-base.zip → чанки + метаданные
│ ├── llm_client.py # chat(), embed(), rerank()
│ ├── prompts.py # SYSTEM, USER, FIX, CRITIC, RECONSIDER
│ ├── config.py # env: LLM_BASE_URL, LLM_API_KEY, DATA_DIR, MAX_WORKERS
│ └── schemas.py # Message, Answer, Source, ParsedQuery, SearchHit
│
└── frontend/ # React + Vite + nginx
├── Dockerfile
├── nginx.conf # listen 3000
├── package.json # react, react-dom, vite
├── package-lock.json
├── vite.config.js # port 3000, dev-proxy /api
├── index.html
├── .dockerignore 
│
└── src/
├── main.jsx # ReactDOM.createRoot(<App />)
├── App.jsx # состояние чата, localStorage, runChat, retry
├── api.js # sendChat, listDocuments, getDocument, ApiError
├── index.css # стили: шапка, сообщения, источники, панели
│
└── components/
├── ChatWindow.jsx # окно чата, форма, ClarifyOptions
├── Message.jsx # пузырь, бейджи, «Повторить», SourceCard
├── SourceCard.jsx # карточка источника → DocumentViewer
├── DocumentViewer.jsx# просмотр документа, подсветка chunk_id
├── DocumentsPanel.jsx# список документов, поиск, группировка
└── ClarifyOptions.jsx# кнопки уточнения
