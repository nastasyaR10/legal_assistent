// src/components/Message.jsx
import SourceCard from "./SourceCard";

export default function Message({ msg, onOpenSource, onRetry, onNewChat }) {
  const isUser = msg.role === "user";
  const answer = msg.answer;

  let statusClass = "";
  let statusLabel = null;

  if (answer?.status === "partial") {
    statusClass = "partial";
    statusLabel = "Ответ неполный";
  } else if (answer?.status === "refusal") {
    statusClass = "refusal";
    statusLabel = "Недостаточно данных";
  } else if (answer?.status === "error") {
    statusClass = "error";
    statusLabel = "Ошибка";
  }

  const isError = answer?.status === "error";
  const canRetry = isError && msg.retryHistory && typeof onRetry === "function";

  const sources = Array.isArray(answer?.sources) ? answer.sources : [];

  return (
    <div className={`message ${isUser ? "user" : "assistant"} ${statusClass}`}>
      {statusLabel && <div className="status-badge">{statusLabel}</div>}
      <div className="content">{msg.content}</div>

      {isError && (
        <div className="error-actions">
          {canRetry && (
            <button className="retry-btn" onClick={() => onRetry(msg.retryHistory)}>
              Повторить
            </button>
          )}
        </div>
      )}

      {sources.length > 0 && (
        <div className="sources">
          <div className="sources-title">Источники:</div>
          {sources.map((src, i) => (
            <SourceCard key={i} source={src} onOpen={() => onOpenSource(src)} />
          ))}
        </div>
      )}
    </div>
  );
}