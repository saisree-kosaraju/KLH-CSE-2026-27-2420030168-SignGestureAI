import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Shared wrapper around window.speechSynthesis.
 *
 *   speak(text)  cancels anything in progress, then speaks `text`
 *   stop()       cancels immediately
 *   speaking     whether an utterance started here is active
 *   supported    whether the browser exposes speech synthesis
 *
 * Prefers an en-IN voice when the browser provides one, otherwise falls back
 * to the browser default. Rate is kept at 0.95.
 */
export function useSpeech() {
  const supported =
    typeof window !== 'undefined' &&
    'speechSynthesis' in window &&
    'SpeechSynthesisUtterance' in window;

  const [speaking, setSpeaking] = useState(false);
  const voiceRef = useRef(null);

  useEffect(() => {
    if (!supported) return undefined;

    const loadVoice = () => {
      const voices = window.speechSynthesis.getVoices() || [];
      voiceRef.current =
        voices.find((v) => v.lang === 'en-IN') ||
        voices.find((v) => (v.lang || '').toLowerCase().startsWith('en-in')) ||
        null;
    };

    loadVoice();
    window.speechSynthesis.addEventListener?.('voiceschanged', loadVoice);
    return () => {
      window.speechSynthesis.removeEventListener?.('voiceschanged', loadVoice);
      window.speechSynthesis.cancel();
    };
  }, [supported]);

  const speak = useCallback(
    (text) => {
      const value = String(text ?? '').trim();
      if (!supported || !value) return;

      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(value);
      utterance.rate = 0.95;
      utterance.pitch = 1;
      utterance.lang = voiceRef.current?.lang || 'en-IN';
      if (voiceRef.current) utterance.voice = voiceRef.current;

      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);

      window.speechSynthesis.speak(utterance);
    },
    [supported],
  );

  const stop = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  return { speak, stop, speaking, supported };
}
