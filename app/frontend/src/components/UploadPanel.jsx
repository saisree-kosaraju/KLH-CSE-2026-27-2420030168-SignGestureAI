import { useCallback, useEffect, useRef, useState } from 'react';
import Alert from './Alert.jsx';
import ProcessingStages from './ProcessingStages.jsx';

const ACCEPT_EXT = ['.mp4', '.mov', '.webm'];
const MAX_BYTES = 200 * 1024 * 1024;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7">
      <path d="M12 15V4m0 0 4 4m-4-4-4 4" />
      <path d="M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" />
    </svg>
  );
}

function CameraGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5">
      <rect x="3" y="6" width="13" height="12" rx="2" />
      <path d="m16 10 5-3v10l-5-3" />
    </svg>
  );
}

export default function UploadPanel({ file, onSelect, onClear, onRecognize, busy, backendOffline }) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const validateAndSelect = useCallback(
    (candidate) => {
      setLocalError(null);
      if (!candidate) return;
      const name = candidate.name.toLowerCase();
      if (!ACCEPT_EXT.some((ext) => name.endsWith(ext))) {
        setLocalError('Unsupported file type. Choose a .mp4, .mov or .webm video.');
        return;
      }
      if (candidate.size === 0) {
        setLocalError('That file is empty. Choose a valid video.');
        return;
      }
      if (candidate.size > MAX_BYTES) {
        setLocalError('That video exceeds the 200 MB limit.');
        return;
      }
      onSelect(candidate);
    },
    [onSelect],
  );

  const openPicker = () => {
    if (!busy) inputRef.current?.click();
  };

  const clearInput = () => {
    if (inputRef.current) inputRef.current.value = '';
  };

  if (busy) {
    return <ProcessingStages />;
  }

  return (
    <>
      <div className="upload">
        <div
          className={`dropzone${dragging ? ' dropzone--drag' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            validateAndSelect(event.dataTransfer.files?.[0]);
          }}
        >
          <span className="dropzone__icon" aria-hidden="true">
            <UploadIcon />
          </span>

          {!file ? (
            <>
              <p className="dropzone__title">Drop a signing video here</p>
              <p className="dropzone__hint">
                Short clips work best. Keep the signer centered and hands visible throughout
                the frame.
              </p>
              <button type="button" className="btn btn--subtle" onClick={openPicker}>
                Browse video
              </button>
              <p className="dropzone__meta">MP4, MOV and WebM · Maximum 200 MB</p>
            </>
          ) : (
            <div className="dropzone__file">
              <p className="dropzone__title">Video ready</p>
              <div className="filemeta">
                <span className="filemeta__name">{file.name}</span>
                <span className="filemeta__size">{formatBytes(file.size)}</span>
              </div>
              <div className="filemeta__actions">
                <button type="button" className="linkbtn" onClick={openPicker}>
                  Replace
                </button>
                <button
                  type="button"
                  className="linkbtn linkbtn--muted"
                  onClick={() => {
                    onClear();
                    clearInput();
                  }}
                >
                  Remove
                </button>
              </div>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm"
            hidden
            onChange={(event) => validateAndSelect(event.target.files?.[0])}
          />
        </div>

        <div className="preview">
          {file && previewUrl ? (
            <video className="preview__video" src={previewUrl} controls playsInline preload="metadata" />
          ) : (
            <div className="preview__empty">
              <CameraGlyph />
              <p>Your selected video will appear here before recognition.</p>
            </div>
          )}
        </div>
      </div>

      {localError ? (
        <Alert tone="warn" onClose={() => setLocalError(null)}>
          {localError}
        </Alert>
      ) : null}

      <div className="workspace-actions">
        {backendOffline ? (
          <span className="workspace-actions__hint">Recognition is disabled while the backend is offline.</span>
        ) : null}
        <button
          type="button"
          className="btn btn--primary"
          disabled={!file || busy || backendOffline}
          onClick={onRecognize}
        >
          Recognize sign <span aria-hidden="true">→</span>
        </button>
      </div>
    </>
  );
}
