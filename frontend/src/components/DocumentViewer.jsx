// src/components/DocumentViewer.jsx
import { useEffect, useRef } from "react";

export default function DocumentViewer({ doc, highlightChunkId, onClose }) {
  const highlightRef = useRef(null);

  useEffect(() => {
    if (highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlightChunkId, doc?.doc_id]);

  const sections = Array.isArray(doc?.sections) ? doc.sections : [];

  return (
    <div className="document-viewer">
      <div className="doc-header">
        <h2>{doc.title}</h2>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      {doc.status === "expired" && (
        <div className="doc-meta">
          <span className="expired-badge">Утратил силу</span>
        </div>
      )}

      <div className="doc-body">
        {sections.length === 0 && (
          <div className="loading-message">В документе нет секций.</div>
        )}
        {sections.map((section) => (
          <div
            key={section.chunk_id}
            ref={section.chunk_id === highlightChunkId ? highlightRef : null}
            className={`doc-section ${section.chunk_id === highlightChunkId ? "highlight" : ""}`}
          >
            <div className="section-label">{section.section}</div>
            <div className="section-text">{section.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}