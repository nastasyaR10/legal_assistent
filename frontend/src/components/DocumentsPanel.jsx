// src/components/DocumentsPanel.jsx
import { useCallback, useEffect, useState } from "react";
import { listDocuments, ApiError } from "../api";

const TYPE_LABELS = {
  contract: "Договоры",
  amendment: "Доп. соглашения",
  regulation: "Регламенты",
  memo: "Служебные записки",
  letter: "Письма",
  law: "Законы",
};

export default function DocumentsPanel({ onOpen, onClose }) {
  const [documents, setDocuments] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError("Не удалось загрузить документы"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const filtered =
    documents?.filter((d) => {
      const q = query.trim().toLowerCase();
      if (!q) return true;
      return (
        (d.title || "").toLowerCase().includes(q) ||
        (TYPE_LABELS[d.doc_type] || "").toLowerCase().includes(q)
      );
    }) || [];

  const groups = {};
  for (const d of filtered) {
    const key = d.doc_type || "other";
    (groups[key] = groups[key] || []).push(d);
  }

  const handleKey = (e, docId) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onOpen(docId);
    }
  };

  return (
    <div className="document-viewer documents-panel">
      <div className="doc-header">
        <h2>База знаний</h2>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      <div className="documents-search">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Поиск по названию или типу..."
        />
      </div>

      <div className="doc-body">
        {loading && <div className="loading-message">Загружаю список...</div>}

        {error && (
          <div className="error-block">
            <div>{error.message}</div>
            {error.retryable && (
              <button className="retry-btn" onClick={load}>Повторить</button>
            )}
          </div>
        )}

        {!loading && !error && filtered.length === 0 && (
          <div className="loading-message">Ничего не найдено.</div>
        )}

        {Object.entries(groups).map(([type, docs]) => (
          <div key={type} className="doc-group">
            <div className="doc-group-title">{TYPE_LABELS[type] || type}</div>
            {docs.map((d) => (
              <div
                key={d.doc_id}
                className="doc-list-item"
                onClick={() => onOpen(d.doc_id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => handleKey(e, d.doc_id)}
              >
                <div className="doc-list-main">
                  <div className="doc-list-title">{d.title}</div>
                  {d.status === "expired" && (
                    <span className="source-expired">Утратил силу</span>
                  )}
                </div>
                <span className="doc-list-arrow" aria-hidden="true">›</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}