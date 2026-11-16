// src/components/ChatWindow.jsx
import { useState } from "react";
import Message from "./Message";
import ClarifyOptions from "./ClarifyOptions";

export default function ChatWindow({
  messages,
  onSend,
  onRetry,
  onNewChat,
  loading,
  onOpenSource,
}) {
  const [input, setInput] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    onSend(text);
    setInput("");
  };

  const lastMessage = messages[messages.length - 1];
  const clarifyOptions =
    lastMessage?.answer?.status === "clarify" && Array.isArray(lastMessage.answer.clarify_options)
      ? lastMessage.answer.clarify_options
      : [];

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2 className="welcome-title">Юридический ассистент</h2>
            <p>Здравствуйте! Я - ваш юридический ассистент.</p>
            <p>Что я умею:</p>
            <ul className="welcome-list">
              <li>Искать положения в договорах, регламентах и законах из базы знаний</li>
              <li>Отвечать со ссылками на конкретные пункты и статьи</li>
              <li>Учитывать редакцию документа, действовавшую на дату события</li>
              <li>Показывать источники</li>
            </ul>
            <p className="welcome-hint">Задайте вопрос: например, об условиях договора, сроках, порядке согласования или норме закона.</p>
          </div>
        )}
        
        {messages.map((msg, i) => (
          <Message
            key={i}
            msg={msg}
            onOpenSource={onOpenSource}
            onRetry={onRetry}
            onNewChat={onNewChat}
          />
        ))}

        {loading && (
          <div className="message assistant loading-message">
            <span className="loading-dots">Ищу в документах...</span>
          </div>
        )}
      </div>

      {clarifyOptions.length > 0 && (
        <ClarifyOptions options={clarifyOptions} onSelect={(msg) => onSend(msg)} />
      )}

      <form onSubmit={handleSubmit} className="input-form">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Задайте вопрос по документам..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Отправить
        </button>
      </form>
    </div>
  );
}

