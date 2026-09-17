import { useEffect, useState } from 'react';
import { useSpeech } from '../hooks/useSpeech.js';

function MetaItem({ label, value }) {
  return (
    <div className="result__meta-item">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export default function PredictionCard({ result, onReset, onAddToText }) {
  const { speak, stop, speaking, supported } = useSpeech();
  const [added, setAdded] = useState(false);

  // A new prediction re-enables "Add to text".
  useEffect(() => {
    setAdded(false);
  }, [result]);

  const percent = (Number(result.confidence) || 0) * 100;
  const sign = result.predicted_sign;

  const handleAdd = () => {
    onAddToText?.(sign);
    setAdded(true);
  };

  return (
    <div className="result">
      <p className="eyebrow result__label">Recognition result</p>

      <div className="result__headline">
        <div>
          <div className="result__sign">{sign}</div>
          <p className="result__confidence">
            <b>{percent.toFixed(2)}%</b> confidence
          </p>
        </div>
        <div className="result__actions">
          <button type="button" className="btn btn--primary btn--sm" onClick={handleAdd} disabled={added}>
            {added ? 'Added to text' : `Add “${sign}” to text`}
          </button>
          {speaking ? (
            <button type="button" className="btn btn--subtle btn--sm" onClick={stop}>
              Stop speaking
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--subtle btn--sm"
              onClick={() => speak(sign)}
              disabled={!supported}
            >
              Speak sign
            </button>
          )}
          <button type="button" className="btn btn--subtle btn--sm" onClick={onReset}>
            Try another video
          </button>
        </div>
      </div>

      <dl className="result__meta">
        <MetaItem label="Class ID" value={<span className="is-mono">{result.predicted_class_id}</span>} />
        <MetaItem
          label="Extraction Status"
          value={String(result.extraction_status || 'unknown').toUpperCase()}
        />
        <MetaItem
          label="Preprocessing Time"
          value={
            <span className="is-mono">
              {Math.round(Number(result.preprocessing_time_ms) || 0)} ms
            </span>
          }
        />
        <MetaItem
          label="Inference Time"
          value={
            <span className="is-mono">
              {(Number(result.inference_time_ms) || 0).toFixed(1)} ms
            </span>
          }
        />
      </dl>

      {!supported ? (
        <p className="processing__foot">Speech synthesis is unavailable in this browser.</p>
      ) : null}
    </div>
  );
}
