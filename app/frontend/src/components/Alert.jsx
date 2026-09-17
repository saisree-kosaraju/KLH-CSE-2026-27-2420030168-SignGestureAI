const GLYPHS = {
  error: '!',
  warn: '!',
  info: 'i',
};

export default function Alert({ tone = 'error', title, children, onClose }) {
  return (
    <div className={`alert alert--${tone}`} role="alert">
      <span className="alert__glyph" aria-hidden="true">
        {GLYPHS[tone] || '!'}
      </span>
      <div className="alert__body">
        {title ? <p className="alert__title">{title}</p> : null}
        <p className="alert__message">{children}</p>
      </div>
      {onClose ? (
        <button type="button" className="alert__close" onClick={onClose} aria-label="Dismiss">
          ×
        </button>
      ) : null}
    </div>
  );
}
