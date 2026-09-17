import { useCallback, useState } from 'react';

const STORAGE_KEY = 'signspeak.recent.v1';

function read() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function write(items) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    /* storage unavailable / quota — recent history is non-critical */
  }
}

/** Keeps the last N successful predictions in localStorage. */
export function useRecentPredictions(limit = 5) {
  const [items, setItems] = useState(read);

  const add = useCallback(
    (entry) => {
      setItems((prev) => {
        const next = [entry, ...prev].slice(0, limit);
        write(next);
        return next;
      });
    },
    [limit],
  );

  const clear = useCallback(() => {
    setItems([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  return { items, add, clear };
}
