const CHIPS = ['MediaPipe', 'PyTorch', 'FastAPI', 'React'];

// A restrained hand-landmark motif: wrist node + five finger chains,
// echoing the MediaPipe topology the pipeline actually uses.
const NODES = [
  [50, 90, 'root'],
  [39, 78], [31, 67], [25, 58],
  [44, 66], [42, 50], [41, 38],
  [51, 63], [52, 45], [53, 31],
  [59, 65], [62, 49], [64, 39],
  [67, 70], [71, 58], [74, 49],
];
const EDGES = [
  [0, 1], [1, 2], [2, 3],
  [0, 4], [4, 5], [5, 6],
  [0, 7], [7, 8], [8, 9],
  [0, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15],
  [1, 4], [4, 7], [7, 10], [10, 13],
];

function LandmarkMotif() {
  return (
    <svg className="hero__motif" viewBox="0 0 100 100" role="img" aria-label="Hand landmark graph">
      {EDGES.map(([a, b], i) => (
        <line
          key={i}
          className="edge"
          x1={NODES[a][0]}
          y1={NODES[a][1]}
          x2={NODES[b][0]}
          y2={NODES[b][1]}
        />
      ))}
      {NODES.map(([x, y, kind], i) => (
        <circle key={i} className={`node${kind ? ' node--root' : ''}`} cx={x} cy={y} r={kind ? 2.6 : 1.8} />
      ))}
    </svg>
  );
}

export default function Hero({ onStart, onHowItWorks }) {
  return (
    <section className="hero">
      <div className="container hero__grid">
        <div>
          <p className="eyebrow eyebrow--accent hero__eyebrow">Visual language, understood</p>
          <h1 className="hero__title">
            <span>Translate Signs</span>
            <span>Into Meaning.</span>
          </h1>
          <p className="hero__lede">
            AI-powered Indian Sign Language recognition using MediaPipe-based landmark
            extraction and deep temporal modeling.
          </p>

          <div className="hero__actions">
            <button type="button" className="btn btn--primary" onClick={onStart}>
              Start recognition <span aria-hidden="true">→</span>
            </button>
            <button type="button" className="btn btn--ghost" onClick={onHowItWorks}>
              How it works
            </button>
          </div>

          <ul className="hero__chips">
            {CHIPS.map((chip) => (
              <li key={chip} className="chip">
                {chip}
              </li>
            ))}
          </ul>
        </div>

        <aside className="hero__aside">
          <p className="eyebrow hero__aside-eyebrow">Built for clarity</p>
          <p className="hero__aside-title">A research-grade recognition pipeline.</p>
          <p className="hero__aside-text">
            A simple, inspectable interface for demos, evaluation, and communicating how the
            model reads movement over time.
          </p>
          <LandmarkMotif />
        </aside>
      </div>
    </section>
  );
}
