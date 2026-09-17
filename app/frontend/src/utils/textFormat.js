/**
 * Deterministic, local sentence assembly for recognized signs.
 *
 * This is NOT sign-language translation and does not attempt grammar. It only
 * joins the recognized sign labels into a readable line. The raw sequence is
 * always kept separately (see `rawSequence`).
 *
 * A future grammar-improvement API could replace `normalizeSentence` without
 * touching the stored raw sequence.
 */

export function cleanWords(words) {
  return (words || [])
    .map((word) => String(word).trim().toLowerCase())
    .filter(Boolean);
}

/** ["i", "want", "water"] -> "I want water." */
export function normalizeSentence(words) {
  const cleaned = cleanWords(words);
  if (cleaned.length === 0) return '';
  const joined = cleaned.join(' ').replace(/\s+/g, ' ').trim();
  return `${joined.charAt(0).toUpperCase()}${joined.slice(1)}.`;
}

/** ["i", "want", "water"] -> ["I", "WANT", "WATER"] (raw, uppercased for display) */
export function rawSequence(words) {
  return (words || [])
    .map((word) => String(word).trim().toUpperCase())
    .filter(Boolean);
}
