export default function OfflineNotice({ onRetry }) {
  return (
    <div className="notice" role="status">
      <div className="notice__body">
        <strong>The inference backend is offline.</strong>
        <p className="notice__text">Start the FastAPI service, then try again.</p>
        <div className="notice__actions">
          <button type="button" className="btn btn--sm" onClick={onRetry}>
            Retry
          </button>
        </div>
      </div>
    </div>
  );
}
