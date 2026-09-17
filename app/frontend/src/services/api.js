/**
 * Single source of truth for talking to the INCLUDE262 FastAPI backend.
 *
 *   GET  /health   -> { status, model_loaded, num_classes }
 *   POST /predict   (multipart: file=<.mp4|.mov>) -> prediction payload
 *
 * The API contract is fixed. This module only reads it.
 */

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(message, { status = 0, kind = 'error' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.kind = kind; // 'error' | 'network' | 'validation'
  }
}

const HTTP_MESSAGES = {
  400: 'The uploaded file is empty. Choose a valid video and try again.',
  413: 'That video is too large. The maximum accepted size is 200 MB.',
  415: 'Unsupported file type. Upload a single .mp4 or .mov video.',
  422: 'This video could not be processed. It may be corrupt, too short, or contain no clearly visible signer.',
  500: 'The recognition model failed while processing this video. Please try again.',
};

export async function getHealth({ signal } = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}/health`, { signal, cache: 'no-store' });
  } catch {
    throw new ApiError('Backend is unreachable.', { kind: 'network' });
  }
  if (!res.ok) {
    throw new ApiError(`Health check failed (${res.status}).`, { status: res.status });
  }
  return res.json();
}

async function postVideo(path, file, { signal } = {}) {
  const form = new FormData();
  form.append('file', file, file.name);

  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, { method: 'POST', body: form, signal });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Confirm the FastAPI server is running.`,
      { kind: 'network' },
    );
  }

  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json())?.detail;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(HTTP_MESSAGES[res.status] || detail || `Request failed (${res.status}).`, {
      status: res.status,
      kind: res.status >= 400 && res.status < 500 ? 'validation' : 'error',
    });
  }

  return res.json();
}

/** Upload-tab recognition — frozen /predict contract, unchanged. */
export async function predictVideo(file, opts = {}) {
  return postVideo('/predict', file, opts);
}

/**
 * Live-camera recognition — /predict/live: active-sign trimming + quality gate +
 * exact training preprocessing + confidence/margin rejection. Returns a payload
 * with `accepted`, `rejection_reason`, `quality`, `trim`, `top2_*` in addition to
 * the usual prediction fields.
 */
export async function predictVideoLive(file, opts = {}) {
  return postVideo('/predict/live', file, opts);
}

/** Supported-vocabulary lookup (closed set of 262 signs). */
export async function getVocab(q, { signal } = {}) {
  const url = new URL(`${API_BASE_URL}/vocab`);
  if (q) url.searchParams.set('q', q);
  let res;
  try {
    res = await fetch(url, { signal, cache: 'no-store' });
  } catch {
    throw new ApiError('Backend is unreachable.', { kind: 'network' });
  }
  if (!res.ok) throw new ApiError(`Vocabulary lookup failed (${res.status}).`, { status: res.status });
  return res.json();
}

/** Dev-only webcam data collector (backend must run with INCLUDE262_COLLECTOR=1). */
export async function saveCollectorSample(blob, fields, { signal } = {}) {
  const form = new FormData();
  const ext = (blob.type || 'video/webm').includes('mp4') ? 'mp4' : 'webm';
  form.append('file', blob, `collect.${ext}`);
  Object.entries(fields).forEach(([k, v]) => form.append(k, v == null ? '' : String(v)));

  let res;
  try {
    res = await fetch(`${API_BASE_URL}/collector/sample`, { method: 'POST', body: form, signal });
  } catch {
    throw new ApiError('Could not reach the collector endpoint.', { kind: 'network' });
  }
  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json())?.detail;
    } catch {
      /* not json */
    }
    throw new ApiError(detail || `Save failed (${res.status}).`, { status: res.status, kind: 'validation' });
  }
  return res.json();
}

export async function getCollectorCounts({ signal } = {}) {
  const res = await fetch(`${API_BASE_URL}/collector/counts`, { signal, cache: 'no-store' });
  if (!res.ok) throw new ApiError(`Counts failed (${res.status}).`, { status: res.status });
  return res.json();
}
