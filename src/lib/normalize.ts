import { FEATURES_PER_HAND, type Hand } from './types';

/**
 * Normalize a single hand into a fixed 63-length feature vector that is
 * invariant to position and scale (but keeps orientation).
 *
 * Steps:
 *  1. Translate so the wrist (landmark 0) is the origin.
 *  2. Scale by the max distance from the wrist to any keypoint.
 *
 * Shared by the static classifier and the dynamic sequence model so both
 * see the same representation.
 */
export function normalizeHand(hand: Hand): Float32Array {
  const out = new Float32Array(FEATURES_PER_HAND);
  if (hand.length === 0) return out;

  const wrist = hand[0];

  // Find max distance from wrist for scale normalization.
  let maxDist = 1e-6;
  for (const p of hand) {
    const dx = p.x - wrist.x;
    const dy = p.y - wrist.y;
    const dz = p.z - wrist.z;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d > maxDist) maxDist = d;
  }

  for (let i = 0; i < hand.length; i++) {
    const p = hand[i];
    out[i * 3] = (p.x - wrist.x) / maxDist;
    out[i * 3 + 1] = (p.y - wrist.y) / maxDist;
    out[i * 3 + 2] = (p.z - wrist.z) / maxDist;
  }
  return out;
}

/**
 * Build the feature vector for a frame given how many hands the language uses.
 *  - 1 hand  -> 63 features (per-hand normalized, same as normalizeHand)
 *  - 2 hands -> 126 features (both hands, SHARED origin/scale so the relative
 *    position of the two hands is preserved — important for BISINDO).
 *
 * For two hands, the (up to 2) detected hands are ordered left-to-right by
 * wrist x so the slot assignment is deterministic in both training and the
 * app. A missing second hand is zero-filled.
 */
export function buildFeatures(hands: Hand[], numHands: 1 | 2): Float32Array {
  if (numHands === 1) {
    return hands.length > 0 ? normalizeHand(hands[0]) : new Float32Array(FEATURES_PER_HAND);
  }

  const out = new Float32Array(FEATURES_PER_HAND * 2);
  if (hands.length === 0) return out;

  // Order hands left-to-right by wrist x (landmark 0).
  const ordered = [...hands].slice(0, 2).sort((a, b) => a[0].x - b[0].x);
  const origin = ordered[0][0];

  let maxDist = 1e-6;
  for (const hand of ordered) {
    for (const p of hand) {
      const dx = p.x - origin.x;
      const dy = p.y - origin.y;
      const dz = p.z - origin.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > maxDist) maxDist = d;
    }
  }

  for (let h = 0; h < ordered.length; h++) {
    const hand = ordered[h];
    const base = h * FEATURES_PER_HAND;
    for (let i = 0; i < hand.length; i++) {
      out[base + i * 3] = (hand[i].x - origin.x) / maxDist;
      out[base + i * 3 + 1] = (hand[i].y - origin.y) / maxDist;
      out[base + i * 3 + 2] = (hand[i].z - origin.z) / maxDist;
    }
  }
  return out;
}

/** Mean per-keypoint displacement between two normalized vectors (motion magnitude). */
export function frameVelocity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i += 3) {
    const dx = a[i] - b[i];
    const dy = a[i + 1] - b[i + 1];
    const dz = a[i + 2] - b[i + 2];
    sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return sum / (a.length / 3);
}
