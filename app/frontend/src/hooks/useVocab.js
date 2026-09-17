import { useCallback, useEffect, useRef, useState } from 'react';
import { getVocab } from '../services/api.js';

/**
 * Debounced lookup against the closed 262-sign INCLUDE vocabulary.
 * Returns { query, setQuery, results, exact, message, loading, error }.
 * `message` is set (to the out-of-vocabulary note) only when nothing matched.
 */
export function useVocab(initial = '') {
  const [query, setQuery] = useState(initial);
  const [state, setState] = useState({
    results: [],
    exact: null,
    message: null,
    loading: false,
    error: null,
  });
  const abortRef = useRef(null);

  const run = useCallback(async (q) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await getVocab(q, { signal: controller.signal });
      setState({
        results: data.results || [],
        exact: data.exact_match || null,
        message: data.message || null,
        loading: false,
        error: null,
      });
    } catch (err) {
      if (err?.name === 'AbortError') return;
      setState((s) => ({ ...s, loading: false, error: 'Vocabulary lookup unavailable.' }));
    }
  }, []);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setState({ results: [], exact: null, message: null, loading: false, error: null });
      return undefined;
    }
    const id = setTimeout(() => run(q), 220);
    return () => clearTimeout(id);
  }, [query, run]);

  useEffect(() => () => abortRef.current?.abort(), []);

  return { query, setQuery, ...state };
}
