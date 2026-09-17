import { useState } from 'react';
import { useVocab } from '../hooks/useVocab.js';

/**
 * Collapsible "Supported Vocabulary" search. The classifier is closed-set
 * (262 INCLUDE signs); this lets you confirm whether an intended sign is covered
 * before trying to record it. Not a redesign — a small utility card.
 */
export default function VocabSearch() {
  const [open, setOpen] = useState(false);
  const { query, setQuery, results, exact, message, loading } = useVocab();

  return (
    <section className="container vocab">
      <button
        type="button"
        className="vocab__head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="vocab__title">Supported vocabulary</span>
        <span className="vocab__meta">
          262 signs · closed set
          <span className={`vocab__caret${open ? ' vocab__caret--open' : ''}`} aria-hidden="true">▾</span>
        </span>
      </button>

      {open ? (
        <div className="vocab__body">
          <input
            type="search"
            className="vocab__input"
            placeholder="Search a sign (e.g. hello, house, thankyou)…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoComplete="off"
          />

          {query.trim() && !loading && !exact && (results.length === 0 || message) ? (
            <p className="vocab__note vocab__note--warn">
              This sign is outside the current 262-sign vocabulary. The model can only
              recognise the signs listed here.
            </p>
          ) : null}

          {exact ? (
            <p className="vocab__note vocab__note--ok">
              <b>{exact.label}</b> is supported (class {exact.class_id}).
            </p>
          ) : null}

          {results.length > 0 ? (
            <ul className="vocab__list">
              {results.map((r) => (
                <li key={r.class_id} className={`vocab__item${r.exact ? ' vocab__item--exact' : ''}`}>
                  <span className="vocab__cid">{r.class_id}</span>
                  <span className="vocab__label">{r.label}</span>
                </li>
              ))}
            </ul>
          ) : null}

          {!query.trim() ? (
            <p className="vocab__note">
              Type to search. Ground-truth labels come from
              <code> metadata/include262_class_mapping_final.csv</code>.
            </p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
