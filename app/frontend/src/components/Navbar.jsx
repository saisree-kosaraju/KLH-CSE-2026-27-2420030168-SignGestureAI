import StatusBadge from './StatusBadge.jsx';

function BrandMark() {
  return (
    <svg className="brand__mark" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <g stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <path d="M4 15c2.4-1.6 2.4-8 4.5-8s2.4 6.4 4.5 8" />
        <path d="M6 8V5.5M15 17.5V15" />
      </g>
    </svg>
  );
}

export default function Navbar({ health, onRetry }) {
  const offline = health.status === 'offline';
  const numClasses = health.payload?.num_classes;

  return (
    <header className="navbar">
      <div className="navbar__inner">
        <div className="brand">
          <BrandMark />
          <span className="brand__text">
            <span className="brand__name">SignSpeak AI</span>
            <span className="brand__sub">Indian Sign Language Recognition</span>
          </span>
        </div>

        <div className="navbar__meta">
          <StatusBadge status={health.status} />
          {offline ? (
            <button type="button" className="navbar__retry" onClick={onRetry}>
              Retry
            </button>
          ) : null}
          <span className="navbar__signs">{numClasses ? `${numClasses} ISL Signs` : '262 ISL Signs'}</span>
        </div>
      </div>
    </header>
  );
}
