import { useCallback, useState } from 'react';

const STORAGE_KEY = 'signspeak.text.v1';

function read() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function write(words) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(words));
  } catch {
    /* storage unavailable / quota — the sentence is a convenience only */
  }
}

/**
 * The raw sequence of recognized signs the user has explicitly added.
 * Nothing is appended automatically — only via `add`.
 */
export function useRecognizedText() {
  const [words, setWords] = useState(read);

  const add = useCallback((word) => {
    const value = String(word ?? '').trim();
    if (!value) return;
    setWords((prev) => {
      const next = [...prev, value];
      write(next);
      return next;
    });
  }, []);

  const undoLast = useCallback(() => {
    setWords((prev) => {
      const next = prev.slice(0, -1);
      write(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setWords([]);
    write([]);
  }, []);

  return { words, add, undoLast, clear };
}
