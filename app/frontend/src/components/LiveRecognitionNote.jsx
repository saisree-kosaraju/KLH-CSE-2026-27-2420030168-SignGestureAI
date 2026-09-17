/**
 * Compact banner for the live-camera (/predict/live) result: acceptance / rejection
 * state, camera-quality guidance, and the active-sign trim summary. Sits above the
 * normal PredictionCard. Never uses alert().
 *
 * Props (one of):
 *   note   = { message, quality, trim, reason }   -> quality block / extraction failure, no prediction
 *   result = { predicted_sign, confidence, live } -> model ran; live has accepted / rejection / quality / trim
 */
export default function LiveRecognitionNote({ note, result, onRetry }) {
  const live = result?.live;
  const quality = note?.quality || live?.quality;
  const trim = note?.trim || live?.trim;

  const blocked = !!note;
  const accepted = !blocked && live?.accepted;
  const uncertain = !blocked && live && !live.accepted;

  const tone = blocked || uncertain ? 'warn' : 'ok';
  const qualityMsgs = quality?.messages || [];

  return (
    <div className={`livenote livenote--${tone}`}>
      <div className="livenote__head">
        <span className="livenote__badge">
          {blocked ? 'Not recognised' : accepted ? 'Recognised' : 'Sign uncertain'}
        </span>
        <span className="livenote__msg">
          {blocked
            ? note.message
            : accepted
              ? `Accepted: “${result.predicted_sign}” at ${(result.confidence * 100).toFixed(1)}% confidence.`
              : (live.message || 'Sign uncertain - please repeat.')}
        </span>
        {onRetry ? (
          <button type="button" className="btn btn--subtle btn--sm livenote__retry" onClick={onRetry}>
            Record again
          </button>
        ) : null}
      </div>

      {uncertain ? (
        <p className="livenote__detail">
          Best guess <b>{result.predicted_sign}</b> ({(result.confidence * 100).toFixed(1)}%),
          next <b>{live.top2_sign}</b> ({(live.top2_confidence * 100).toFixed(1)}%),
          margin {(live.top2_margin * 100).toFixed(1)}%.
          {' '}The model only recognises 262 signs; if you signed something else it cannot be recognised.
        </p>
      ) : null}

      {qualityMsgs.length ? (
        <ul className="livenote__list">
          {qualityMsgs.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      ) : null}

      {trim && trim.total_frames ? (
        <p className="livenote__trim">
          Trimmed to the active sign: frames {trim.trimmed_frame_range?.[0]}–{trim.trimmed_frame_range?.[1]}
          {' '}of {trim.total_frames}
          {trim.original_duration_s != null && trim.trimmed_duration_s != null
            ? ` (${trim.original_duration_s.toFixed(1)}s → ${trim.trimmed_duration_s.toFixed(1)}s)`
            : ''}
          {trim.motion_score != null ? ` · motion ${trim.motion_score}` : ''}
          {trim.trim_fallback ? ` · kept full clip (${trim.trim_fallback})` : ''}
        </p>
      ) : null}
    </div>
  );
}
