function formatTimestamp(ts) {
  try {
    return new Date(ts).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return '';
  }
}

export default function RecentPredictions({ items, onClear }) {
  return (
    <section className="container history-section">
      <div className="card">
        <div className="card__headrow">
          <div>
            <p className="eyebrow">Local history</p>
            <h3 className="card__title" style={{ marginTop: '6px' }}>
              Recent predictions
            </h3>
          </div>
          {items.length > 0 ? (
            <button type="button" className="btn btn--subtle" onClick={onClear}>
              Clear history
            </button>
          ) : null}
        </div>

        {items.length === 0 ? (
          <p className="empty-state">
            Successful predictions will stay here on this device for quick reference.
          </p>
        ) : (
          <ul className="recents">
            {items.map((item) => (
              <li key={item.timestamp} className="recent">
                <span className="recent__sign">{item.sign}</span>
                <span className="recent__conf">
                  {((Number(item.confidence) || 0) * 100).toFixed(1)}%
                </span>
                <span className="recent__time">{formatTimestamp(item.timestamp)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
