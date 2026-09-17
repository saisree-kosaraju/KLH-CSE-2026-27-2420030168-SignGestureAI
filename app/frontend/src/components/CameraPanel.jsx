import { useRef, useState } from 'react';
import Alert from './Alert.jsx';
import ProcessingStages from './ProcessingStages.jsx';
import LiveLandmarkData from './LiveLandmarkData.jsx';
import { useCameraRecorder } from '../hooks/useCameraRecorder.js';
import { useLiveLandmarks } from '../hooks/useLiveLandmarks.js';

const LAYER_META = [
  { key: 'hands', label: 'Hands' },
  { key: 'pose', label: 'Pose' },
  { key: 'face', label: 'Face' },
];

function formatClock(ms) {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function CameraGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5">
      <rect x="3" y="6" width="13" height="12" rx="2" />
      <path d="m16 10 5-3v10l-5-3" />
    </svg>
  );
}

export default function CameraPanel({ onRecognize, busy, backendOffline }) {
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

  const toggleLayer = (key) => setLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  const submitRecording = async () => {
    const file = await cam.getRecordedFile('camera-recording');
    if (!file) {
      cam.setNotice({ tone: 'error', message: 'The recorded clip could not be read for upload.' });
      return;
    }
    onRecognize(file);
  };

  if (busy) {
    return <ProcessingStages />;
  }

  const { phase, elapsed, liveActive } = cam;
  const statusLabel =
    phase === 'recording'
      ? `Recording · ${formatClock(elapsed)}`
      : phase === 'ready' || phase === 'recorded'
        ? 'Camera Ready'
        : 'Camera Off';

  return (
    <>
      <div className="camera">
        <div className="camera__side">
          <span className={`camera__status camera__status--${phase}`}>
            <span className="camera__status-dot" aria-hidden="true" />
            {statusLabel}
          </span>

          <ul className="camera__guidance">
            <li>Keep your upper body and both hands visible.</li>
            <li>Perform one sign per recording.</li>
            <li>Use a well-lit background.</li>
          </ul>

          {liveActive ? (
            <div className="camera__layers" role="group" aria-label="Overlay layers">
              {LAYER_META.map(({ key, label }) => (
                <button
                  key={key}
                  type="button"
                  className={`camera__layer${layers[key] ? ' camera__layer--on' : ''}`}
                  aria-pressed={layers[key]}
                  onClick={() => toggleLayer(key)}
                >
                  {label}
                </button>
              ))}
            </div>
          ) : null}

          <div className="camera__controls">
            {phase === 'idle' ? (
              <button type="button" className="btn btn--primary" onClick={cam.startCamera}>
                Start camera
              </button>
            ) : null}

            {phase === 'ready' ? (
              <>
                <button type="button" className="btn btn--primary" onClick={cam.startRecording}>
                  Record sign
                </button>
                <button type="button" className="btn btn--subtle" onClick={cam.closeCamera}>
                  Stop camera
                </button>
              </>
            ) : null}

            {phase === 'recording' ? (
              <button type="button" className="btn btn--danger" onClick={cam.stopRecording}>
                Stop recording
              </button>
            ) : null}

            {phase === 'recorded' ? (
              <>
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={submitRecording}
                  disabled={backendOffline}
                >
                  Recognize sign <span aria-hidden="true">→</span>
                </button>
                <button type="button" className="btn btn--subtle" onClick={cam.discardRecording}>
                  Re-record
                </button>
              </>
            ) : null}
          </div>

          {phase === 'recording' ? (
            <p className="camera__hint">Recommended clip length: 2-5 seconds. Idle padding is trimmed automatically.</p>
          ) : null}
          {phase === 'recorded' && backendOffline ? (
            <p className="camera__hint">Recognition is disabled while the backend is offline.</p>
          ) : null}
        </div>

        <div className="camera__stage">
          {phase === 'recorded' && cam.recordedUrl ? (
            <video className="camera__video camera__video--playback" src={cam.recordedUrl} controls playsInline />
          ) : (
            <video ref={cam.videoRef} className="camera__video" autoPlay muted playsInline />
          )}

          {liveActive ? <canvas ref={canvasRef} className="camera__overlay" aria-hidden="true" /> : null}

          {phase === 'idle' ? (
            <div className="camera__placeholder">
              <CameraGlyph />
              <p>Start the camera to preview and record a sign.</p>
            </div>
          ) : null}

          {phase === 'recording' ? (
            <span className="camera__rec">
              <span className="camera__rec-dot" aria-hidden="true" />
              REC {formatClock(elapsed)}
            </span>
          ) : null}

          {liveActive && landmarks.status === 'loading' ? (
            <span className="camera__mp camera__mp--loading">Loading hand &amp; pose tracking…</span>
          ) : null}
          {liveActive && landmarks.status === 'ready' ? (
            <span className="camera__mp">
              {landmarks.fps || 0} FPS ·{' '}
              {landmarks.leftHand?.detected || landmarks.rightHand?.detected
                ? `${(landmarks.leftHand?.detected ? 1 : 0) + (landmarks.rightHand?.detected ? 1 : 0)} hand(s)`
                : 'no hands'}
            </span>
          ) : null}
        </div>
      </div>

      {liveActive && landmarks.status === 'error' ? (
        <p className="camera__mp-error">{landmarks.error}</p>
      ) : null}

      {liveActive ? (
        <LiveLandmarkData
          snapshot={landmarks}
          phase={phase}
          elapsed={elapsed}
          onShowCoordsChange={setCoordsOpen}
        />
      ) : null}

      {cam.notice ? (
        <Alert tone={cam.notice.tone} onClose={() => cam.setNotice(null)}>
          {cam.notice.message}
        </Alert>
      ) : null}
    </>
  );
}
