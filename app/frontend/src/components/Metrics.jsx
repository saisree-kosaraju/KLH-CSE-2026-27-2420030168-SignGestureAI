const METRICS = [
  { value: '98.04%', label: 'Test Accuracy' },
  { value: '99.51%', label: 'Top-5 Accuracy' },
  { value: '262', label: 'ISL Classes' },
];

export default function Metrics() {
  return (
    <section className="metrics">
      <div className="container metrics__grid">
        {METRICS.map((metric) => (
          <div key={metric.label} className="metric">
            <span className="metric__value">{metric.value}</span>
            <span className="metric__label">{metric.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
