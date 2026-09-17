import { useCallback, useEffect, useRef, useState } from 'react';
import { getHealth } from '../services/api.js';

/**
 * Polls GET /health on an interval.
 * status: 'checking' | 'online' | 'loading' | 'offline'
 */
export function useHealth(intervalMs = 10000) {
  const [status, setStatus] = useState('checking');
  const [payload, setPayload] = useState(null);
  const mounted = useRef(true);

  const check = useCallback(async () => {
    try {
      const data = await getHealth();
      if (!mounted.current) return;
      setPayload(data);
      setStatus(data?.model_loaded ? 'online' : 'loading');
    } catch {
      if (!mounted.current) return;
      setPayload(null);
      setStatus('offline');
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    check();
    const id = setInterval(check, intervalMs);
    return () => {
      mounted.current = false;
      clearInterval(id);
    };
  }, [check, intervalMs]);

  return { status, payload, refresh: check };
}
