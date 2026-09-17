const STATES = {
  checking: { text: 'Checking backend', tone: 'neutral' },
  online: { text: 'Model Online', tone: 'ok' },
  loading: { text: 'Model Loading', tone: 'warn' },
  offline: { text: 'Backend Offline', tone: 'err' },
};

export default function StatusBadge({ status }) {
  const state = STATES[status] || STATES.checking;
  return (
    <span className={`status status--${state.tone}`} title={`Backend: ${state.text}`}>
      <span className="status__dot" aria-hidden="true" />
      {state.text}
    </span>
  );
}
