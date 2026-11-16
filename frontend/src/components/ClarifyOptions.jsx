// src/components/ClarifyOptions.jsx
export default function ClarifyOptions({ options, onSelect }) {
  return (
    <div className="clarify-options">
      <div className="clarify-title">Уточните, пожалуйста:</div>
      {options.map((opt, i) => (
        <button key={i} className="clarify-button" onClick={() => onSelect(opt.message)}>
          {opt.label}
        </button>
      ))}
    </div>
  );
}