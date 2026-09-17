import { useEffect, useState } from 'react';

/* =========================================================================
   Live Landmark Data — compact, expandable panel for the live camera.
   All values here are browser-side MediaPipe visualisation/debug feedback.
   They are NOT the backend's exact 469-D geometry vector: the model still
   runs the authoritative 534-D landmark / 469-D geometry / 4-D availability
   / 64-frame preprocessing on the server after "Recognize sign".
   ========================================================================= */

function StatusRow({ label, detected }) {
  return (
    <div className="lld__status-row">
      <span>{label}</span>
      <span className={`lld__chip lld__chip--${detected ? 'on' : 'off'}`}>
        {detected ? 'Detected' : 'Not detected'}
      </span>
    </div>
  );
}

function TrackRow({ label, state }) {
  return (
    <div className="lld__status-row">
      <span>{label}</span>
      <span className={`lld__chip lld__chip--${state === 'Tracking' ? 'on' : state === 'Ready' ? 'idle' : 'off'}`}>
        {state}
      </span>
    </div>
  );
}

function num(v) {
  return v == null ? '—' : typeof v === 'number' ? v.toFixed(3) : v;
}

export default function LiveLandmarkData({ snapshot, phase, elapsed, onShowCoordsChange }) {
  const [open, setOpen] = useState(true);
  const [showCoords, setShowCoords] = useState(false);
  const [selected, setSelected] = useState('right');

  useEffect(() => {
    onShowCoordsChange?.(showCoords);
  }, [showCoords, onShowCoordsChange]);

  const s = snapshot || {};
  const g = s.geometry || {};
  const rec = s.recordingStats;
  const last = s.lastRecording;

  const handsState = s.leftHand?.detected || s.rightHand?.detected ? 'Tracking' : 'Not detected';
  const poseState = s.pose?.detected ? 'Tracking' : 'Not detected';
  const geoState = g.palmScale != null ? 'Tracking' : 'Not detected';
  const motionState = g.handMotion != null && g.handMotion > 0 ? 'Tracking' : 'Ready';

  const hand = selected === 'left' ? s.leftHand : s.rightHand;
  const coords = hand?.landmarks;

  return (
    <section className="lld">
      <button type="button" className="lld__head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="lld__title">Live Landmark Data</span>
        <span className="lld__meta">
          {s.status === 'ready'
            ? `${s.fps || 0} FPS`
            : s.status === 'loading'
              ? 'loading…'
              : s.status === 'error'
                ? 'unavailable'
                : 'idle'}
          <span className={`lld__caret${open ? ' lld__caret--open' : ''}`} aria-hidden="true">
            ▾
          </span>
        </span>
      </button>

      {s.error ? <p className="lld__error">{s.error}</p> : null}

      {open ? (
        <div className="lld__body">
          <div className="lld__grid">
            {/* ---- detection status ---- */}
            <div className="lld__block">
              <h4 className="lld__block-title">Detection</h4>
              <StatusRow label="Left hand" detected={!!s.leftHand?.detected} />
              <StatusRow label="Right hand" detected={!!s.rightHand?.detected} />
              <StatusRow label="Pose" detected={!!s.pose?.detected} />
              <StatusRow label="Face" detected={!!s.face?.detected} />
              <p className="lld__note">
                {(s.leftHand?.detected ? 21 : 0) + (s.rightHand?.detected ? 21 : 0)} hand landmarks ·{' '}
                {s.pose?.points || 0} pose · {s.face?.points || 0} face
              </p>
            </div>

            {/* ---- preprocessing status ---- */}
            <div className="lld__block">
              <h4 className="lld__block-title">Live Preprocessing Status</h4>
              <TrackRow label="Hands tracking" state={handsState} />
              <TrackRow label="Pose tracking" state={poseState} />
              <TrackRow label="Geometry tracking" state={geoState} />
              <TrackRow label="Motion tracking" state={motionState} />
              {phase === 'recording' ? (
                <p className="lld__note">Recording duration: {(elapsed / 1000).toFixed(1)}s</p>
              ) : null}
            </div>

            {/* ---- live geometry ---- */}
            <div className="lld__block">
              <h4 className="lld__block-title">Live Geometry</h4>
              <div className="lld__kv">
                <span>Palm scale</span>
                <span>{num(g.palmScale)}</span>
                <span>Wrist position</span>
                <span>
                  {g.wrist ? `${num(g.wrist.x)}, ${num(g.wrist.y)}, ${num(g.wrist.z)}` : '—'}
                </span>
                <span>Finger spread</span>
                <span>{num(g.fingerSpread)}</span>
                <span>Hand motion</span>
                <span>{num(g.handMotion)}</span>
                <span>Hand velocity</span>
                <span>{num(g.handVelocity)}</span>
                <span>Hand availability</span>
                <span>{g.availability || '—'}</span>
              </div>
              <p className="lld__note">Live debug values — not the backend 469-D geometry vector.</p>
            </div>
          </div>

          {/* ---- recording coverage ---- */}
          {(rec || last) ? (
            <div className="lld__rec">
              {rec ? (
                <>
                  <span>Captured frames: <b>{rec.frames}</b></span>
                  <span>Hand coverage: <b>{rec.handCoverage}%</b></span>
                  <span>Pose coverage: <b>{rec.poseCoverage}%</b></span>
                  <span className="lld__note">observed live (browser)</span>
                </>
              ) : (
                <>
                  <span>Last clip — frames: <b>{last.frames}</b></span>
                  <span>Hand coverage: <b>{last.handCoverage}%</b></span>
                  <span>Pose coverage: <b>{last.poseCoverage}%</b></span>
                  <span className="lld__note">
                    browser estimate · authoritative coverage appears after recognition
                  </span>
                </>
              )}
            </div>
          ) : null}

          {/* ---- coordinates ---- */}
          <div className="lld__coords-head">
            <label className="lld__toggle">
              <input type="checkbox" checked={showCoords} onChange={(e) => setShowCoords(e.target.checked)} />
              Show coordinates
            </label>
            {showCoords ? (
              <div className="lld__seg" role="group" aria-label="Selected hand">
                <button
                  type="button"
                  className={`lld__seg-btn${selected === 'left' ? ' is-active' : ''}`}
                  onClick={() => setSelected('left')}
                >
                  Left
                </button>
                <button
                  type="button"
                  className={`lld__seg-btn${selected === 'right' ? ' is-active' : ''}`}
                  onClick={() => setSelected('right')}
                >
                  Right
                </button>
              </div>
            ) : null}
          </div>

          {showCoords ? (
            <div className="lld__table-wrap">
              {coords ? (
                <table className="lld__table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>X</th>
                      <th>Y</th>
                      <th>Z</th>
                    </tr>
                  </thead>
                  <tbody>
                    {coords.map((p, i) => (
                      <tr key={i}>
                        <td>{i}</td>
                        <td>{p.x.toFixed(3)}</td>
                        <td>{p.y.toFixed(3)}</td>
                        <td>{p.z.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="lld__note lld__note--pad">
                  {selected === 'left' ? 'Left' : 'Right'} hand not detected.
                </p>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
