const MODALITIES = [
  { key: 'left_hand', label: 'Left hand' },
  { key: 'right_hand', label: 'Right hand' },
  { key: 'pose', label: 'Pose' },
  { key: 'face', label: 'Face' },
];

function toneFor(percent) {
  if (percent == null) return 'muted';
  if (percent >= 85) return 'ok';
  if (percent >= 50) return 'warn';
  return 'err';
}

export default function DetectionQuality({ coverage }) {
  const availability = coverage?.per_frame_availability;
  if (!availability) return null;

  return (
    <div className="card">
      <h3 className="card__title">Landmark detection quality</h3>
      <p className="card__subtitle">Share of the 64 sampled frames where each modality was detected.</p>
      <div className="quality">
        {MODALITIES.map(({ key, label }) => {
          const raw = availability[key];
          const percent = typeof raw === 'number' ? Math.round(raw * 100) : null;
          const tone = toneFor(percent);
          return (
            <div key={key} className="quality__item">
              <div className="quality__row">
                <span className="quality__label">{label}</span>
                <span className={`quality__pct quality__pct--${tone}`}>
                  {percent == null ? '—' : `${percent}%`}
                </span>
              </div>
              <span className="quality__track">
                <span
                  className={`quality__fill quality__fill--${tone}`}
                  style={{ width: `${percent ?? 0}%` }}
                />
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
