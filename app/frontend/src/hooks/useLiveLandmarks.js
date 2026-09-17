import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  FaceLandmarker,
} from '@mediapipe/tasks-vision';

/* =========================================================================
   Browser-side MediaPipe engine for the LIVE CAMERA visualization only.
   This never touches the FastAPI /predict flow — the backend still runs the
   authoritative INCLUDE262 preprocessing (534-D landmarks, 469-D geometry,
   4-D availability mask, 64-frame sampling). Everything here is live
   visual/debug feedback rendered straight onto a canvas overlay.
   ========================================================================= */

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const MODEL = {
  hands:
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
  pose:
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  face:
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
};

// ~30 FPS visualisation cap. Real cadence is whatever detection can sustain
// (typically 15–30 FPS), which is exactly the range we want.
const FRAME_INTERVAL_MS = 33;
const SNAPSHOT_MS = 200; // low-frequency React state updates only
const MAX_DPR = 2;

// Official MediaPipe upper-body pose subset: shoulders, elbows, wrists (+ hips
// for a restrained torso frame). Indices are the standard 33-point topology.
const UPPER_POSE = new Set([11, 12, 13, 14, 15, 16, 23, 24]);
const FINGER_TIPS = [4, 8, 12, 16, 20];

const STYLE = {
  hand: { point: '#2f6fed', pointEdge: 'rgba(255,255,255,0.9)', line: 'rgba(120,170,250,0.75)' },
  pose: { point: '#64748b', pointEdge: 'rgba(255,255,255,0.85)', line: 'rgba(100,116,139,0.5)' },
  face: { line: 'rgba(120,150,200,0.16)', oval: 'rgba(120,150,200,0.4)' },
};

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function centroid(pts) {
  let x = 0;
  let y = 0;
  for (const p of pts) {
    x += p.x;
    y += p.y;
  }
  return { x: x / pts.length, y: y / pts.length };
}

const EMPTY = {
  status: 'idle',
  error: null,
  fps: 0,
  leftHand: { detected: false, landmarks: null },
  rightHand: { detected: false, landmarks: null },
  pose: { detected: false, points: 0 },
  face: { detected: false, points: 0 },
  geometry: null,
  recordingStats: null,
  lastRecording: null,
};

/**
 * @param {object}   opts
 * @param {React.RefObject<HTMLVideoElement>} opts.videoRef
 * @param {React.RefObject<HTMLCanvasElement>} opts.canvasRef
 * @param {boolean}  opts.active      camera stream is live and previewing
 * @param {boolean}  opts.recording   MediaRecorder is currently running
 * @param {boolean}  opts.mirrored    preview is CSS-mirrored (overlay must match)
 * @param {{hands:boolean,pose:boolean,face:boolean}} opts.layers
 * @param {boolean}  opts.wantCoords  coordinate table is open (copy hand points into state)
 */
export function useLiveLandmarks({
  videoRef,
  canvasRef,
  active,
  recording,
  mirrored = true,
  layers,
  wantCoords = false,
}) {
  const [snapshot, setSnapshot] = useState(EMPTY);

  const filesetRef = useRef(null);
  const detectorsRef = useRef({ hands: null, pose: null, face: null });
  const initTokenRef = useRef(0);
  const rafRef = useRef(0);
  const lastFrameAtRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const fpsRef = useRef(0);

  // High-frequency values live in refs; only a throttled copy reaches state.
  const liveRef = useRef({ ...EMPTY });
  const motionRef = useRef({ prev: null, ema: 0, vel: 0 });
  const recRef = useRef(null);
  const optsRef = useRef({ layers, recording, mirrored, wantCoords, active });
  optsRef.current = { layers, recording, mirrored, wantCoords, active };

  const statusRef = useRef('idle');
  const setStatus = useCallback((status, error = null) => {
    statusRef.current = status;
    liveRef.current.status = status;
    liveRef.current.error = error;
    setSnapshot((s) => (s.status === status && s.error === error ? s : { ...liveRef.current }));
  }, []);

  /* ---- detector lifecycle -------------------------------------------------- */

  const ensureDetectors = useCallback(async () => {
    const token = initTokenRef.current;
    const need = optsRef.current.layers;
    try {
      if (!filesetRef.current) {
        setStatus('loading');
        filesetRef.current = await FilesetResolver.forVisionTasks(WASM_PATH);
      }
      if (token !== initTokenRef.current) return;

      const make = async (kind, factory, options) => {
        if (!need[kind] || detectorsRef.current[kind]) return;
        try {
          detectorsRef.current[kind] = await factory.createFromOptions(filesetRef.current, {
            baseOptions: { modelAssetPath: MODEL[kind], delegate: 'GPU' },
            runningMode: 'VIDEO',
            ...options,
          });
        } catch {
          detectorsRef.current[kind] = await factory.createFromOptions(filesetRef.current, {
            baseOptions: { modelAssetPath: MODEL[kind], delegate: 'CPU' },
            runningMode: 'VIDEO',
            ...options,
          });
        }
      };

      await make('hands', HandLandmarker, { numHands: 2, minHandDetectionConfidence: 0.5 });
      if (token !== initTokenRef.current) return;
      await make('pose', PoseLandmarker, { numPoses: 1 });
      if (token !== initTokenRef.current) return;
      await make('face', FaceLandmarker, { numFaces: 1 });
      if (token !== initTokenRef.current) return;

      setStatus('ready');
    } catch (err) {
      if (token !== initTokenRef.current) return;
      setStatus(
        'error',
        'Live landmark tracking could not start (MediaPipe failed to load). Recording and recognition still work.',
      );
    }
  }, [setStatus]);

  /* ---- drawing ----------------------------------------------------------- */

  const drawConnections = (ctx, pts, connections, map, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (const c of connections) {
      const a = pts[c.start];
      const b = pts[c.end];
      if (!a || !b) continue;
      const pa = map(a);
      const pb = map(b);
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
    }
    ctx.stroke();
  };

  const drawPoints = (ctx, pts, map, fill, edge, radius, only) => {
    ctx.lineWidth = Math.max(1, radius * 0.4);
    ctx.strokeStyle = edge;
    ctx.fillStyle = fill;
    pts.forEach((p, i) => {
      if (only && !only.has(i)) return;
      const { x, y } = map(p);
      const r = FINGER_TIPS.includes(i) && !only ? radius * 1.25 : radius;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  };

  /* ---- geometry (browser-side live/debug values, NOT the 469-D vector) --- */

  const computeGeometry = (left, right, fps) => {
    const hand = right || left;
    const availability = left && right ? 'L + R' : left ? 'Left only' : right ? 'Right only' : 'None';
    if (!hand) {
      motionRef.current.prev = null;
      return {
        palmScale: null,
        wrist: null,
        fingerSpread: null,
        handMotion: 0,
        handVelocity: 0,
        availability,
      };
    }
    const wrist = hand[0];
    const palmScale = dist(wrist, hand[9]) || 1e-6;

    let spread = 0;
    for (let i = 1; i < FINGER_TIPS.length; i += 1) {
      spread += dist(hand[FINGER_TIPS[i - 1]], hand[FINGER_TIPS[i]]);
    }
    spread = spread / palmScale;

    const c = centroid(hand);
    const m = motionRef.current;
    let step = 0;
    if (m.prev) step = Math.hypot(c.x - m.prev.x, c.y - m.prev.y);
    m.prev = c;
    m.ema = m.ema * 0.7 + step * 0.3;
    m.vel = step * (fps || 0);

    return {
      palmScale: Number(palmScale.toFixed(4)),
      wrist: { x: Number(wrist.x.toFixed(3)), y: Number(wrist.y.toFixed(3)), z: Number((wrist.z || 0).toFixed(3)) },
      fingerSpread: Number(spread.toFixed(3)),
      handMotion: Number(m.ema.toFixed(4)),
      handVelocity: Number(m.vel.toFixed(3)),
      availability,
    };
  };

  /* ---- main loop -------------------------------------------------------- */

  useEffect(() => {
    if (!active) return undefined;

    initTokenRef.current += 1;
    ensureDetectors();

    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return;

      const now = performance.now();
      if (now - lastFrameAtRef.current < FRAME_INTERVAL_MS) return;
      const delta = now - lastFrameAtRef.current;
      lastFrameAtRef.current = now;
      if (delta < 1000) fpsRef.current = fpsRef.current * 0.8 + (1000 / delta) * 0.2;

      // Match the canvas backing store to the displayed video box.
      const cssW = video.clientWidth || canvas.clientWidth;
      const cssH = video.clientHeight || canvas.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      const bw = Math.round(cssW * dpr);
      const bh = Math.round(cssH * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }

      // object-fit: cover mapping — the video is scaled to fill and centre-cropped.
      const scale = Math.max(bw / video.videoWidth, bh / video.videoHeight);
      const renderW = video.videoWidth * scale;
      const renderH = video.videoHeight * scale;
      const offX = (bw - renderW) / 2;
      const offY = (bh - renderH) / 2;
      const map = (p) => ({ x: offX + p.x * renderW, y: offY + p.y * renderH });

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, bw, bh);

      const { layers: L, wantCoords: WC, recording: REC } = optsRef.current;
      const ready = statusRef.current === 'ready';
      let left = null;
      let right = null;
      let poseFound = false;
      let faceFound = false;

      if (ready && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const ts = now;
        const d = detectorsRef.current;

        try {
          if (L.hands && d.hands) {
            const r = d.hands.detectForVideo(video, ts);
            (r.landmarks || []).forEach((lm, i) => {
              const label = r.handedness?.[i]?.[0]?.categoryName || (i === 0 ? 'Right' : 'Left');
              if (label === 'Left') left = lm;
              else right = lm;
            });
          }
          if (L.pose && d.pose) {
            const r = d.pose.detectForVideo(video, ts);
            const lm = r.landmarks?.[0];
            if (lm) {
              poseFound = true;
              drawConnections(
                ctx,
                lm,
                PoseLandmarker.POSE_CONNECTIONS.filter(
                  (c) => UPPER_POSE.has(c.start) && UPPER_POSE.has(c.end),
                ),
                map,
                STYLE.pose.line,
                2 * dpr,
              );
              drawPoints(ctx, lm, map, STYLE.pose.point, STYLE.pose.pointEdge, 3 * dpr, UPPER_POSE);
            }
          }
          if (L.face && d.face) {
            const r = d.face.detectForVideo(video, ts);
            const lm = r.faceLandmarks?.[0];
            if (lm) {
              faceFound = true;
              drawConnections(ctx, lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, map, STYLE.face.line, 0.6 * dpr);
              drawConnections(ctx, lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, map, STYLE.face.oval, 1 * dpr);
            }
          }
        } catch {
          /* transient detector error on a frame — skip it */
        }
      } else if (ready) {
        // same video frame: keep last hand refs so the overlay doesn't flicker
        left = liveRef.current.leftHand.landmarks;
        right = liveRef.current.rightHand.landmarks;
      }

      // Draw hands last so they sit on top.
      for (const lm of [left, right]) {
        if (!lm) continue;
        drawConnections(ctx, lm, HandLandmarker.HAND_CONNECTIONS, map, STYLE.hand.line, 2 * dpr);
        drawPoints(ctx, lm, map, STYLE.hand.point, STYLE.hand.pointEdge, 3.4 * dpr);
      }

      // ---- update live refs -------------------------------------------------
      const fps = Math.round(fpsRef.current);
      const geometry = computeGeometry(left, right, fps);
      const pack = (lm) =>
        lm && WC ? lm.map((p) => ({ x: p.x, y: p.y, z: p.z || 0 })) : null;

      liveRef.current = {
        status: statusRef.current,
        error: liveRef.current.error,
        fps,
        leftHand: { detected: !!left, landmarks: pack(left) },
        rightHand: { detected: !!right, landmarks: pack(right) },
        pose: { detected: poseFound, points: poseFound ? UPPER_POSE.size : 0 },
        face: { detected: faceFound, points: faceFound ? 468 : 0 },
        geometry,
        recordingStats: null,
        lastRecording: recRef.current?.frozen || null,
      };

      // ---- recording coverage (browser-observed, live) --------------------
      if (REC) {
        if (!recRef.current || recRef.current.frozen) {
          recRef.current = { frames: 0, hand: 0, pose: 0, startedAt: now, frozen: null };
        }
        const rc = recRef.current;
        rc.frames += 1;
        if (left || right) rc.hand += 1;
        if (poseFound) rc.pose += 1;
        liveRef.current.recordingStats = {
          frames: rc.frames,
          durationMs: now - rc.startedAt,
          handCoverage: rc.frames ? Math.round((rc.hand / rc.frames) * 100) : 0,
          poseCoverage: rc.frames ? Math.round((rc.pose / rc.frames) * 100) : 0,
        };
      } else if (recRef.current && !recRef.current.frozen) {
        const rc = recRef.current;
        rc.frozen = {
          frames: rc.frames,
          durationMs: now - rc.startedAt,
          handCoverage: rc.frames ? Math.round((rc.hand / rc.frames) * 100) : 0,
          poseCoverage: rc.frames ? Math.round((rc.pose / rc.frames) * 100) : 0,
        };
        liveRef.current.lastRecording = rc.frozen;
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    const snapId = setInterval(() => setSnapshot({ ...liveRef.current }), SNAPSHOT_MS);

    return () => {
      cancelAnimationFrame(rafRef.current);
      clearInterval(snapId);
      lastVideoTimeRef.current = -1;
      const cv = canvasRef.current;
      if (cv) cv.getContext('2d')?.clearRect(0, 0, cv.width, cv.height);
    };
  }, [active, ensureDetectors, canvasRef, videoRef]);

  // React to layer changes while the camera is already running.
  useEffect(() => {
    if (active) ensureDetectors();
  }, [active, layers.hands, layers.pose, layers.face, ensureDetectors]);

  // Full teardown of the (heavy) detectors when the camera is closed / unmounts.
  useEffect(
    () => () => {
      initTokenRef.current += 1;
      const d = detectorsRef.current;
      ['hands', 'pose', 'face'].forEach((k) => {
        try {
          d[k]?.close();
        } catch {
          /* ignore */
        }
        d[k] = null;
      });
      filesetRef.current = null;
      statusRef.current = 'idle';
    },
    [],
  );

  return snapshot;
}
