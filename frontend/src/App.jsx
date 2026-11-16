// src/App.jsx
import { useEffect, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import DocumentViewer from "./components/DocumentViewer";
import DocumentsPanel from "./components/DocumentsPanel";
import { getDocument, sendChat, ApiError } from "./api";

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDoc, setOpenDoc] = useState(null);
  const [highlightChunkId, setHighlightChunkId] = useState(null);
  const [documentsOpen, setDocumentsOpen] = useState(false);

  // восстановление истории
  useEffect(() => {
    try {
      const raw = localStorage.getItem("legal-assistant:messages");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setMessages(parsed);
      }
    } catch {}
  }, []);

  // сохранение истории
  useEffect(() => {
    try {
      localStorage.setItem("legal-assistant:messages", JSON.stringify(messages));
    } catch {}
  }, [messages]);

  const runChat = async (history) => {
    setLoading(true);
    try {
      const answer = await sendChat(history);
      setMessages([...history, { role: "assistant", content: answer.text, answer }]);
    } catch (e) {
      const isApi = e instanceof ApiError;
      const message = isApi ? e.message : `Непредвиденная ошибка: ${e?.message || e}`;
      const retryable = isApi ? e.retryable : false;
      setMessages([
        ...history,
        {
          role: "assistant",
          content: message,
          answer: { status: "error", sources: [], clarify_options: [] },
          retryHistory: retryable ? history : null,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = (text) => {
    if (!text.trim() || loading) return;
    const userMessage = { role: "user", content: text };
    runChat([...messages, userMessage]);
  };

  const handleRetry = (history) => {
    if (!history) return;
    runChat(history);
  };

  const handleNewChat = () => {
    setMessages([]);
    setOpenDoc(null);
    setHighlightChunkId(null);
    try {
      localStorage.removeItem("legal-assistant:messages");
    } catch {}
  };

  const handleOpenSource = async (source) => {
    try {
      const doc = await getDocument(source.doc_id);
      setOpenDoc(doc);
      setHighlightChunkId(source.chunk_id);
    } catch (e) {
      alert(e.message || "Не удалось открыть документ");
    }
  };

  const handleOpenDocumentById = async (docId) => {
    try {
      const doc = await getDocument(docId);
      setOpenDoc(doc);
      setHighlightChunkId(null);
      setDocumentsOpen(false);
    } catch (e) {
      alert(e.message || "Не удалось открыть документ");
    }
  };

  return (
    <div className="app">
      <div className="chat-panel">
        <header className="app-header">
          <div className="header-left">
            <h1>Lawyer Assistant</h1>
            <span className="app-subtitle">Помощник юриста ПСБ</span>
          </div>
          <div className="header-actions">
            <button className="header-btn" onClick={() => setDocumentsOpen(true)}>
              Документы
            </button>
            <button
              className="header-btn"
              onClick={handleNewChat}
              disabled={messages.length === 0 && !openDoc}
            >
              Новый чат
            </button>
          </div>
        </header>

        <ChatWindow
          messages={messages}
          onSend={handleSend}
          onRetry={handleRetry}
          onNewChat={handleNewChat}
          loading={loading}
          onOpenSource={handleOpenSource}
        />
      </div>

      {openDoc && (
        <DocumentViewer
          doc={openDoc}
          highlightChunkId={highlightChunkId}
          onClose={() => {
            setOpenDoc(null);
            setHighlightChunkId(null);
          }}
        />
      )}

      {documentsOpen && (
        <DocumentsPanel
          onOpen={handleOpenDocumentById}
          onClose={() => setDocumentsOpen(false)}
        />
      )}
    </div>
  );
}

export default App;