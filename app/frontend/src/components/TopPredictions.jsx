export default function TopPredictions({ items }) {
  if (!Array.isArray(items) || items.length === 0) return null;

  const maxProbability = Math.max(...items.map((item) => Number(item.probability) || 0), 1e-4);

  return (
    <div className="card">
      <h3 className="card__title">Top predictions</h3>
      <ul className="bars">
        {items.map((item, index) => {
          const probability = Number(item.probability) || 0;
          const percent = probability * 100;
          const width = Math.max((probability / maxProbability) * 100, 1.5);
          return (
            <li key={`${item.class_id}-${index}`} className={`bar${index === 0 ? ' bar--lead' : ''}`}>
              <span className="bar__name" title={item.sign}>
                {item.sign}
              </span>
              <span className="bar__track">
                <span className="bar__fill" style={{ width: `${width}%` }} />
              </span>
              <span className="bar__value">
                {percent < 0.01 ? '<0.01%' : `${percent.toFixed(2)}%`}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
