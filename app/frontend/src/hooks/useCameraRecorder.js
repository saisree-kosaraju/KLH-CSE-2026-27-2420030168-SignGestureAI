import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Webcam capture + MediaRecorder lifecycle, shared by the Live Camera tab and the
 * dev-only data collector. Keeps recording separate from tracking / recognition
 * (Module 20). The recorded blob is the RAW getUserMedia stream — never the
 * CSS-mirrored preview — so backend inference orientation stays correct.
 *
 * phase: 'idle' | 'ready' | 'recording' | 'recorded'
 */

const RECORD_PREFERENCES = [
  'video/mp4',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

const MIN_RECORD_MS = 1200;
const MAX_RECORD_MS = 8000;
const TICK_MS = 100;

function extensionFor(mimeBase) {
  if (mimeBase === 'video/mp4') return 'mp4';
  if (mimeBase === 'video/quicktime') return 'mov';
  return 'webm';
}

export function useCameraRecorder() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const startedAtRef = useRef(0);
  const lastBlobRef = useRef(null);
  const lastDurationRef = useRef(0);

  const [phase, setPhase] = useState('idle');
  const [elapsed, setElapsed] = useState(0);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [recordedType, setRecordedType] = useState(null);
  const [notice, setNotice] = useState(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(
    () => () => {
      clearTimer();
      try {
        if (recorderRef.current && recorderRef.current.state !== 'inactive') {
          recorderRef.current.stop();
        }
      } catch {
        /* ignore */
      }
      stopStream();
    },
    [clearTimer, stopStream],
  );

  useEffect(() => {
    if (recordedUrl) return () => URL.revokeObjectURL(recordedUrl);
    return undefined;
  }, [recordedUrl]);

  const liveActive = phase === 'ready' || phase === 'recording';

  // The live <video> is swapped for a playback <video> in 'recorded' — re-bind
  // the stream and resume playback whenever we return to a live phase.
  useEffect(() => {
    if (!liveActive) return;
    const video = videoRef.current;
    if (video && streamRef.current && video.srcObject !== streamRef.current) {
      video.srcObject = streamRef.current;
    }
    video?.play?.().catch(() => {});
  }, [liveActive, phase]);

  const startCamera = useCallback(async () => {
    setNotice(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setNotice({ tone: 'error', message: 'This browser does not provide camera access.' });
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setPhase('ready');
    } catch (err) {
      const name = err?.name || '';
      let message = 'The camera could not be started.';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        message = 'Camera permission was denied. Allow camera access in your browser, then start again.';
      } else if (['NotFoundError', 'OverconstrainedError', 'NotReadableError'].includes(name)) {
        message = 'No usable camera was found. Check that a webcam is connected and not in use by another app.';
      }
      setNotice({ tone: 'error', message });
    }
  }, []);

  const closeCamera = useCallback(() => {
    clearTimer();
    stopStream();
    setPhase('idle');
    setElapsed(0);
  }, [clearTimer, stopStream]);

  const finalizeRecording = useCallback(() => {
    clearTimer();
    const duration = Date.now() - startedAtRef.current;
    const type = recorderRef.current?.mimeType || 'video/webm';
    const blob = new Blob(chunksRef.current, { type });
    chunksRef.current = [];

    if (blob.size < 1024 || duration < MIN_RECORD_MS) {
      setPhase('ready');
      setElapsed(0);
      setNotice({
        tone: 'warn',
        message:
          blob.size < 1024
            ? 'The recording came back empty. Try again and hold the sign for a couple of seconds.'
            : 'That recording was too short. Aim for roughly 2-5 seconds with one clear sign.',
      });
      return;
    }

    lastBlobRef.current = blob;
    lastDurationRef.current = duration;
    setRecordedUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(blob);
    });
    setRecordedType(blob.type);
    setPhase('recorded');
  }, [clearTimer]);

  const stopRecording = useCallback(() => {
    try {
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }
    } catch {
      clearTimer();
      setPhase('ready');
    }
  }, [clearTimer]);

  const startRecording = useCallback(() => {
    setNotice(null);
    if (!streamRef.current || typeof window.MediaRecorder === 'undefined') {
      setNotice({ tone: 'error', message: 'Recording is not supported in this browser.' });
      return;
    }
    const mimeType = RECORD_PREFERENCES.find((t) => window.MediaRecorder.isTypeSupported?.(t));
    let recorder;
    try {
      recorder = new MediaRecorder(streamRef.current, mimeType ? { mimeType } : undefined);
    } catch {
      setNotice({ tone: 'error', message: 'This browser could not start a supported recording format.' });
      return;
    }

    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = finalizeRecording;
    recorder.onerror = () => {
      clearTimer();
      setPhase('ready');
      setElapsed(0);
      setNotice({ tone: 'error', message: 'Recording failed. Please try again.' });
    };

    try {
      recorder.start();
    } catch {
      setNotice({ tone: 'error', message: 'Recording could not be started.' });
      return;
    }

    recorderRef.current = recorder;
    startedAtRef.current = Date.now();
    setElapsed(0);
    setPhase('recording');

    timerRef.current = setInterval(() => {
      const ms = Date.now() - startedAtRef.current;
      setElapsed(ms);
      if (ms >= MAX_RECORD_MS) stopRecording();
    }, TICK_MS);
  }, [clearTimer, finalizeRecording, stopRecording]);

  const discardRecording = useCallback(() => {
    lastBlobRef.current = null;
    setRecordedUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setRecordedType(null);
    setElapsed(0);
    setPhase(streamRef.current ? 'ready' : 'idle');
  }, []);

  /** The recorded clip as a File (base mime, sane extension), or null. */
  const getRecordedFile = useCallback(async (name = 'camera-recording') => {
    const blob = lastBlobRef.current;
    if (!blob) return null;
    const base = (blob.type || 'video/webm').split(';')[0].trim();
    return new File([blob], `${name}.${extensionFor(base)}`, { type: base });
  }, []);

  return {
    videoRef,
    streamRef,
    phase,
    elapsed,
    recordedUrl,
    recordedType,
    recordedDurationMs: lastDurationRef.current,
    notice,
    setNotice,
    liveActive,
    startCamera,
    closeCamera,
    startRecording,
    stopRecording,
    discardRecording,
    getRecordedFile,
  };
}
