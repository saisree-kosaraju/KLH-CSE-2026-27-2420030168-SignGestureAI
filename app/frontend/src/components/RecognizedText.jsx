import { useState } from 'react';
import { normalizeSentence, rawSequence } from '../utils/textFormat.js';
import { useSpeech } from '../hooks/useSpeech.js';

export default function RecognizedText({ words, onUndo, onClear }) {
  const { speak, stop, speaking, supported } = useSpeech();
  const [copied, setCopied] = useState(false);

  const sentence = normalizeSentence(words);
  const raw = rawSequence(words);
  const hasWords = raw.length > 0;

  const copyText = async () => {
    if (!sentence) return;
    try {
      await navigator.clipboard.writeText(sentence);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section className="container recognized-text">
      <div className="card">
        <div className="card__headrow">
          <div>
            <p className="eyebrow">Communication aid</p>
            <h3 className="card__title rtext__heading">Recognized Text</h3>
          </div>
        </div>

        {!hasWords ? (
          <p className="empty-state">
            Recognize a sign, then choose “Add to text”. Nothing is added automatically, so a
            wrong prediction never enters your sentence.
          </p>
        ) : (
          <>
            <p className="rtext__sentence">{sentence}</p>
            <div className="rtext__raw">
              <span className="rtext__raw-label">Raw signs</span>
              <span className="rtext__chips">
                {raw.map((word, index) => (
                  <span key={`${word}-${index}`} className="rtext__chip">
                    {word}
                  </span>
                ))}
              </span>
            </div>
          </>
        )}

        <div className="rtext__actions">
          <button type="button" className="btn btn--subtle" onClick={onUndo} disabled={!hasWords}>
            Undo last
          </button>
          <button type="button" className="btn btn--subtle" onClick={onClear} disabled={!hasWords}>
            Clear text
          </button>
          <button type="button" className="btn btn--subtle" onClick={copyText} disabled={!sentence}>
            {copied ? 'Copied' : 'Copy text'}
          </button>
          {speaking ? (
            <button type="button" className="btn btn--subtle" onClick={stop}>
              Stop speaking
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => speak(sentence)}
              disabled={!sentence || !supported}
            >
              Speak text
            </button>
          )}
        </div>

        {!supported ? (
          <p className="processing__foot">Speech synthesis is unavailable in this browser.</p>
        ) : null}
      </div>
    </section>
  );
}
