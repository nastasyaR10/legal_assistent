// src/components/SourceCard.jsx
export default function SourceCard({ source, onOpen }) {
  return (
    <div className="source-card" onClick={onOpen}>
      <div className="source-title">{source.doc_title}</div>
      <div className="source-section">{source.section}</div>
      <div className="source-quote">«{source.quote}»</div>
      {source.doc_status === "expired" && (
        <div className="source-expired">Утратил силу</div>
      )}
    </div>
  );
}