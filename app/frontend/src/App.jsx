import { useCallback, useMemo, useRef, useState } from 'react';
import Navbar from './components/Navbar.jsx';
import Hero from './components/Hero.jsx';
import Metrics from './components/Metrics.jsx';
import UploadPanel from './components/UploadPanel.jsx';
import CameraPanel from './components/CameraPanel.jsx';
import OfflineNotice from './components/OfflineNotice.jsx';
import Alert from './components/Alert.jsx';
import PredictionCard from './components/PredictionCard.jsx';
import TopPredictions from './components/TopPredictions.jsx';
import DetectionQuality from './components/DetectionQuality.jsx';
import RecentPredictions from './components/RecentPredictions.jsx';
import RecognizedText from './components/RecognizedText.jsx';
import HowItWorks from './components/HowItWorks.jsx';
import Footer from './components/Footer.jsx';
import VocabSearch from './components/VocabSearch.jsx';
import LiveRecognitionNote from './components/LiveRecognitionNote.jsx';
import CollectorPanel from './components/CollectorPanel.jsx';
import { useHealth } from './hooks/useHealth.js';
import { useRecentPredictions } from './hooks/useRecentPredictions.js';
import { useRecognizedText } from './hooks/useRecognizedText.js';
import { ApiError, predictVideo, predictVideoLive, API_BASE_URL } from './services/api.js';

const COLLECTOR_MODE = import.meta.env.VITE_COLLECTOR === '1';

/** Adapt the /predict/live payload to the shape the result components expect. */
function adaptLiveResult(data) {
  return {
    predicted_class_id: data.predicted_class_id,
    predicted_sign: data.predicted_sign,
    confidence: data.confidence,
    top5: data.top5 || [],
    modality_coverage: data.modality_coverage,
    extraction_status: data.extraction_status,
    preprocessing_time_ms: data.timings_ms?.extract,
    inference_time_ms: data.timings_ms?.inference,
    live: {
      accepted: data.accepted,
      rejection_reason: data.rejection_reason,
      message: data.message,
      top2_sign: data.top2_sign,
      top2_confidence: data.top2_confidence,
      top2_margin: data.top2_margin,
      thresholds: data.thresholds,
      quality: data.quality,
      trim: data.trim,
    },
  };
}

const TABS = [
  {
    id: 'upload',
    label: 'Upload video',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 15V4m0 0 4 4m-4-4-4 4" />
        <path d="M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" />
      </svg>
    ),
  },
  {
    id: 'camera',
    label: 'Live camera',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="6" width="13" height="12" rx="2" />
        <path d="m16 10 5-3v10l-5-3" />
      </svg>
    ),
  },
  ...(COLLECTOR_MODE
    ? [
        {
          id: 'collect',
          label: 'Data collection',
          icon: (
            <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          ),
        },
      ]
    : []),
];

export default function App() {
  const health = useHealth();
  const recent = useRecentPredictions(5);
  const text = useRecognizedText();

  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [liveNote, setLiveNote] = useState(null); // quality-block / extraction rejection, no prediction

  const workspaceRef = useRef(null);
  const howItWorksRef = useRef(null);
  const resultRef = useRef(null);

  const backendOffline = health.status === 'offline';

  const devHost = useMemo(() => {
    if (!import.meta.env.DEV) return null;
    try {
      return new URL(API_BASE_URL).host;
    } catch {
      return null;
    }
  }, []);

  const scrollTo = (ref) =>
    ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const handleRecognize = useCallback(
    async (videoFile) => {
      const target = videoFile || file;
      if (!target || busy) return;
      const live = activeTab === 'camera';

      setBusy(true);
      setError(null);
      setResult(null);
      setLiveNote(null);

      try {
        if (live) {
          const data = await predictVideoLive(target);
          if (data.stage !== 'complete') {
            // quality block or extraction failure — no prediction to show
            setLiveNote({
              message: data.message || 'The recording could not be recognised.',
              quality: data.quality,
              trim: data.trim,
              reason: data.rejection_reason,
            });
            return;
          }
          const adapted = adaptLiveResult(data);
          setResult(adapted);
          if (data.accepted) {
            recent.add({ sign: data.predicted_sign, confidence: data.confidence, timestamp: Date.now() });
          }
          requestAnimationFrame(() =>
            resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
          );
          return;
        }

        const data = await predictVideo(target);
        setResult(data);
        recent.add({ sign: data.predicted_sign, confidence: data.confidence, timestamp: Date.now() });
        requestAnimationFrame(() =>
          resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        );
      } catch (err) {
        if (err?.name === 'AbortError') return;
        const message =
          err instanceof ApiError
            ? err.message
            : 'Something went wrong while contacting the backend.';
        setError({ title: 'Recognition failed', message });
      } finally {
        setBusy(false);
      }
    },
    [busy, file, recent, activeTab],
  );

  const handleReset = useCallback(() => {
    setResult(null);
    setError(null);
    setLiveNote(null);
    setFile(null);
    requestAnimationFrame(() => scrollTo(workspaceRef));
  }, []);

  return (
    <div className="app">
      <Navbar health={health} onRetry={health.refresh} />

      <main>
        <Hero
          onStart={() => scrollTo(workspaceRef)}
          onHowItWorks={() => scrollTo(howItWorksRef)}
        />

        <Metrics />

        <section className="container workspace-section" ref={workspaceRef}>
          <div className="section-head workspace-section__head">
            <div>
              <p className="eyebrow">Recognition workspace</p>
              <h2 className="section-head__title">Give the model a moment of movement.</h2>
            </div>
            <span className="section-note">MP4, MOV or WebM · Maximum 200 MB</span>
          </div>

          <div className="panel">
            <div className="panel__tabs" role="tablist" aria-label="Recognition input">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={activeTab === tab.id}
                  className={`tab${activeTab === tab.id ? ' tab--active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
              {devHost ? <span className="panel__devhost">{devHost}</span> : null}
            </div>

            <div className="panel__body">
              {backendOffline ? <OfflineNotice onRetry={health.refresh} /> : null}

              {error ? (
                <Alert tone="error" title={error.title} onClose={() => setError(null)}>
                  {error.message}
                </Alert>
              ) : null}

              {activeTab === 'upload' ? (
                <UploadPanel
                  file={file}
                  onSelect={setFile}
                  onClear={() => setFile(null)}
                  onRecognize={() => handleRecognize()}
                  busy={busy}
                  backendOffline={backendOffline}
                />
              ) : activeTab === 'collect' ? (
                <CollectorPanel backendOffline={backendOffline} />
              ) : (
                <CameraPanel
                  onRecognize={handleRecognize}
                  busy={busy}
                  backendOffline={backendOffline}
                />
              )}
            </div>
          </div>
        </section>

        <div ref={resultRef} />

        {liveNote ? (
          <section className="container results-section">
            <LiveRecognitionNote note={liveNote} onRetry={handleReset} />
          </section>
        ) : null}

        {result ? (
          <section className="container results-section">
            {result.live ? <LiveRecognitionNote result={result} onRetry={handleReset} /> : null}
            <PredictionCard result={result} onReset={handleReset} onAddToText={text.add} />
            <div className="results__grid">
              <TopPredictions items={result.top5} />
              <DetectionQuality coverage={result.modality_coverage} />
            </div>
          </section>
        ) : null}

        <RecognizedText words={text.words} onUndo={text.undoLast} onClear={text.clear} />

        <RecentPredictions items={recent.items} onClear={recent.clear} />

        <VocabSearch />

        <HowItWorks ref={howItWorksRef} />
      </main>

      <Footer />
    </div>
  );
}
