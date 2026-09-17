import { forwardRef } from 'react';

const METRICS = [
  { value: '64', label: 'frames sampled' },
  { value: '534-D', label: 'landmark representation' },
  { value: '469-D', label: 'geometry representation' },
  { value: '4-D', label: 'availability mask' },
  { value: '262', label: 'sign classes' },
];

const STEPS = [
  {
    n: '01',
    title: 'Video input',
    body: 'A short MP4 or MOV clip is uniformly sampled into 64 temporal frames.',
  },
  {
    n: '02',
    title: 'Landmark extraction',
    body: 'MediaPipe tracks hands, pose and selected facial landmarks into a 534-D landmark representation.',
  },
  {
    n: '03',
    title: 'Geometry features',
    body: 'Motion and spatial relationships create a 469-D geometry representation.',
  },
  {
    n: '04',
    title: 'Availability mask',
    body: 'A 4-D availability mask tells the model which modalities were visible in each frame.',
  },
  {
    n: '05',
    title: 'Part-aware temporal encoding',
    body: 'Part-specific Conv1D encoders process left hand, right hand, pose, face and geometry features.',
  },
  {
    n: '06',
    title: 'HTMA temporal recognition',
    body: 'The PyTorch HTMA model performs temporal modeling and ranks 262 ISL sign classes.',
  },
];

const HowItWorks = forwardRef(function HowItWorks(_props, ref) {
  return (
    <section className="how" ref={ref}>
      <div className="how__inner">
        <div>
          <p className="eyebrow eyebrow--accent">Under the hood</p>
          <h2 className="how__title">How SignSpeak AI Works</h2>
          <p className="how__desc">
            A compact multi-stream representation gives the model enough context to read
            movement, geometry and landmark availability over time.
          </p>

          <div className="how__metrics">
            {METRICS.map((metric) => (
              <div key={metric.label} className="how__metric">
                <span className="how__metric-value">{metric.value}</span>
                <span className="how__metric-label">{metric.label}</span>
              </div>
            ))}
          </div>

          <div className="how__accuracy">
            <span className="how__accuracy-value">98.04%</span>
            <span className="how__accuracy-label">Final test accuracy</span>
          </div>
        </div>

        <ol className="pipeline">
          {STEPS.map((step) => (
            <li key={step.n} className="pipeline__step">
              <span className="pipeline__num">{step.n}</span>
              <div>
                <h4>{step.title}</h4>
                <p>{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
});

export default HowItWorks;
