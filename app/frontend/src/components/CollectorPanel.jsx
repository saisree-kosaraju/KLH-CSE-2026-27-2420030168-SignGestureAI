import { useCallback, useEffect, useRef, useState } from 'react';
import Alert from './Alert.jsx';
import LiveLandmarkData from './LiveLandmarkData.jsx';
import { useCameraRecorder } from '../hooks/useCameraRecorder.js';
import { useLiveLandmarks } from '../hooks/useLiveLandmarks.js';
import { useVocab } from '../hooks/useVocab.js';
import { saveCollectorSample, getCollectorCounts } from '../services/api.js';

/**
 * DEV-ONLY webcam data collector (Module 10). Mounted only when VITE_COLLECTOR=1
 * and the backend runs with INCLUDE262_COLLECTOR=1.
 *
 * The HUMAN picks the label before recording — the classifier is never consulted.
 * Metadata (session / background / lighting / position / distance / clothing) is
 * captured so webcam validation can be split by environment (Modules 11, 15).
 */

const OPTIONS = {
  background_id: ['plain_wall', 'normal_room', 'furniture_visible', 'cluttered_room', 'bright_bg', 'dark_bg'],
  lighting: ['daylight', 'indoor_normal', 'indoor_dim', 'side_light'],
  position: ['center', 'slightly_left', 'slightly_right', 'near', 'far'],
  camera_distance: ['close', 'normal', 'far'],
  clothing: ['light', 'dark', 'patterned'],
};

const LAYERS = [
  { key: 'hands', label: 'Hands' },
  { key: 'pose', label: 'Pose' },
  { key: 'face', label: 'Face' },
];

function Select({ label, value, onChange, options }) {
  return (
    <label className="collector__field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o.replace(/_/g, ' ')}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function CollectorPanel({ backendOffline }) {
  const cam = useCameraRecorder();
  const canvasRef = useRef(null);
  const [layers, setLayers] = useState({ hands: true, pose: true, face: false });
  const [coordsOpen, setCoordsOpen] = useState(false);

  const landmarks = useLiveLandmarks({
    videoRef: cam.videoRef,
    canvasRef,
    active: cam.liveActive,
    recording: cam.phase === 'recording',
    mirrored: true,
    layers,
    wantCoords: coordsOpen,
  });

  const vocab = useVocab();
  const [selected, setSelected] = useState(null); // { class_id, label }
  const [split, setSplit] = useState('train');
  const [meta, setMeta] = useState({
    session_id: `s${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`,
    background_id: OPTIONS.background_id[0],
    lighting: OPTIONS.lighting[0],
    position: OPTIONS.position[0],
    camera_distance: OPTIONS.camera_distance[1],
    clothing: OPTIONS.clothing[0],
    notes: '',
  });
  const [counts, setCounts] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState(null);

  const refreshCounts = useCallback(() => {
    getCollectorCounts()
      .then(setCounts)
      .catch(() => setCounts(null));
  }, []);

  useEffect(() => {
    refreshCounts();
  }, [refreshCounts]);

  const setMetaField = (k, v) => setMeta((m) => ({ ...m, [k]: v }));

  const save = async () => {
    if (!selected) {
      setSaveMsg({ tone: 'warn', text: 'Select a sign first.' });
      return;
    }
    const file = await cam.getRecordedFile('collect');
    if (!file) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const snap = landmarks || {};
      const res = await saveCollectorSample(file, {
        class_id: selected.class_id,
        label: selected.label,
        split,
        session_id: meta.session_id,
        background_id: meta.background_id,
        lighting: meta.lighting,
        position: meta.position,
        camera_distance: meta.camera_distance,
        clothing: meta.clothing,
        notes: meta.notes,
        duration_s: (cam.recordedDurationMs / 1000).toFixed(2),
        fps: snap.fps || '',
        left_coverage: snap.lastRecording?.handCoverage != null ? snap.lastRecording.handCoverage / 100 : '',
        right_coverage: '',
        pose_coverage: snap.lastRecording?.poseCoverage != null ? snap.lastRecording.poseCoverage / 100 : '',
        face_coverage: '',
        app_version: 'signspeak-ai-collector',
      });
      setSaveMsg({ tone: 'ok', text: res.message || `Saved ${selected.label} to ${split}.` });
      cam.discardRecording();
      refreshCounts();
    } catch (err) {
      setSaveMsg({ tone: 'error', text: err?.message || 'Save failed.' });
    } finally {
      setSaving(false);
    }
  };

  const classCount = selected && counts?.by_class?.[String(selected.class_id)];

  return (
    <div className="collector">
      <div className="collector__note">
        Dev-only collection tool. You pick the label; it is stored as ground truth.
        Clips are saved to <code>webcam_data/{split}/</code>.
      </div>

      {/* ---- class picker ---- */}
      <div className="collector__pick">
        <input
          type="search"
          className="collector__search"
          placeholder="Search a supported sign…"
          value={vocab.query}
          onChange={(e) => vocab.setQuery(e.target.value)}
          autoComplete="off"
        />
        {vocab.results.length > 0 ? (
          <ul className="collector__results">
            {vocab.results.slice(0, 8).map((r) => (
              <li key={r.class_id}>
                <button
                  type="button"
                  className={`collector__result${selected?.class_id === r.class_id ? ' is-active' : ''}`}
                  onClick={() => setSelected({ class_id: r.class_id, label: r.label })}
                >
                  <span className="collector__cid">{r.class_id}</span>
                  {r.label}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        {vocab.query.trim() && !vocab.loading && vocab.results.length === 0 ? (
          <p className="collector__oov">Outside the 262-sign vocabulary.</p>
        ) : null}
      </div>

      {selected ? (
        <div className="collector__selected">
          Recording label: <b>{selected.label}</b> (class {selected.class_id})
          {classCount ? (
            <span className="collector__tally">
              collected — train {classCount.train || 0} · val {classCount.val || 0}
            </span>
          ) : null}
        </div>
      ) : null}

      {/* ---- metadata ---- */}
      <div className="collector__meta">
        <label className="collector__field">
          <span>session_id</span>
          <input value={meta.session_id} onChange={(e) => setMetaField('session_id', e.target.value)} />
        </label>
        {Object.entries(OPTIONS).map(([k, opts]) => (
          <Select key={k} label={k} value={meta[k]} options={opts} onChange={(v) => setMetaField(k, v)} />
        ))}
        <div className="collector__split">
          <span>split</span>
          <div className="lld__seg">
            {['train', 'val'].map((s) => (
              <button
                key={s}
                type="button"
                className={`lld__seg-btn${split === s ? ' is-active' : ''}`}
                onClick={() => setSplit(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ---- camera + overlay ---- */}
      <div className="camera">
        <div className="camera__side">
          <span className={`camera__status camera__status--${cam.phase}`}>
            <span className="camera__status-dot" aria-hidden="true" />
            {cam.phase === 'recording' ? 'Recording' : cam.phase === 'idle' ? 'Camera Off' : 'Camera Ready'}
          </span>

          {cam.liveActive ? (
            <div className="camera__layers" role="group" aria-label="Overlay layers">
              {LAYERS.map(({ key, label }) => (
                <button
                  key={key}
                  type="button"
                  className={`camera__layer${layers[key] ? ' camera__layer--on' : ''}`}
                  aria-pressed={layers[key]}
                  onClick={() => setLayers((p) => ({ ...p, [key]: !p[key] }))}
                >
                  {label}
                </button>
              ))}
            </div>
          ) : null}

          <div className="camera__controls">
            {cam.phase === 'idle' ? (
              <button type="button" className="btn btn--primary" onClick={cam.startCamera}>
                Start camera
              </button>
            ) : null}
            {cam.phase === 'ready' ? (
              <>
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={cam.startRecording}
                  disabled={!selected}
                >
                  Record sign
                </button>
                <button type="button" className="btn btn--subtle" onClick={cam.closeCamera}>
                  Stop camera
                </button>
              </>
            ) : null}
            {cam.phase === 'recording' ? (
              <button type="button" className="btn btn--danger" onClick={cam.stopRecording}>
                Stop recording
              </button>
            ) : null}
            {cam.phase === 'recorded' ? (
              <>
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={save}
                  disabled={saving || backendOffline}
                >
                  {saving ? 'Saving…' : `Save to ${split}`}
                </button>
                <button type="button" className="btn btn--subtle" onClick={cam.discardRecording}>
                  Re-record
                </button>
              </>
            ) : null}
          </div>

          {!selected ? <p className="camera__hint">Pick a sign to enable recording.</p> : null}
          {saveMsg ? (
            <p className={`collector__savemsg collector__savemsg--${saveMsg.tone}`}>{saveMsg.text}</p>
          ) : null}
          {counts ? (
            <p className="camera__hint">
              Total collected: {counts.total} (train {counts.by_split.train} · val {counts.by_split.val})
            </p>
          ) : null}
        </div>

        <div className="camera__stage">
          {cam.phase === 'recorded' && cam.recordedUrl ? (
            <video className="camera__video camera__video--playback" src={cam.recordedUrl} controls playsInline />
          ) : (
            <video ref={cam.videoRef} className="camera__video" autoPlay muted playsInline />
          )}
          {cam.liveActive ? <canvas ref={canvasRef} className="camera__overlay" aria-hidden="true" /> : null}
          {cam.phase === 'recording' ? (
            <span className="camera__rec">
              <span className="camera__rec-dot" aria-hidden="true" />
              REC
            </span>
          ) : null}
          {cam.liveActive && landmarks.status === 'ready' ? (
            <span className="camera__mp">{landmarks.fps || 0} FPS</span>
          ) : null}
        </div>
      </div>

      {cam.liveActive && landmarks.status === 'error' ? (
        <p className="camera__mp-error">{landmarks.error}</p>
      ) : null}

      {cam.liveActive ? (
        <LiveLandmarkData
          snapshot={landmarks}
          phase={cam.phase}
          elapsed={cam.elapsed}
          onShowCoordsChange={setCoordsOpen}
        />
      ) : null}

      {cam.notice ? (
        <Alert tone={cam.notice.tone} onClose={() => cam.setNotice(null)}>
          {cam.notice.message}
        </Alert>
      ) : null}
    </div>
  );
}
