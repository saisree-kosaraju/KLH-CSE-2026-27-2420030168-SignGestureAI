import { useEffect, useState } from 'react';

const STAGES = [
  'Preparing video',
  'Extracting MediaPipe landmarks',
  'Building motion and geometry features',
  'Running recognition model',
  'Generating prediction',
];

/**
 * Sequential, non-quantitative activity indicator. The backend call is a
 * single request — this only loops through the real pipeline stages until
 * the response resolves. No fabricated progress percentages.
 */
export default function ProcessingStages() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setActive((current) => (current + 1) % STAGES.length);
    }, 1500);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="processing">
      <p className="processing__title">
        <span className="spinner" aria-hidden="true" />
        Analyzing your sign…
      </p>

      <ol className="pstages">
        {STAGES.map((stage, index) => (
          <li
            key={stage}
            className={`pstage${index === active ? ' pstage--active' : ''}${
              index < active ? ' pstage--done' : ''
            }`}
          >
            <span className="pstage__dot" aria-hidden="true" />
            <span>{stage}</span>
          </li>
        ))}
      </ol>

      <p className="processing__foot">
        MediaPipe processes all 64 sampled frames — this usually takes a few seconds.
      </p>
    </div>
  );
}
