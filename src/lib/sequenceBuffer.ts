import { FEATURES_PER_HAND } from './types';

/**
 * Resample a variable-length motion segment to a fixed number of timesteps so
 * the dynamic model always sees the same input shape [TIMESTEPS, F]. Feature
 * length F is inferred from the frames (63 for one hand, 126 for two), so this
 * works for both single- and two-handed languages.
 */
export function resampleSequence(frames: Float32Array[], timesteps: number): Float32Array {
  const featLen = frames[0]?.length ?? FEATURES_PER_HAND;
  const out = new Float32Array(timesteps * featLen);
  if (frames.length === 0) return out;

  for (let i = 0; i < timesteps; i++) {
    const srcIdx =
      frames.length === 1 ? 0 : Math.round((i / (timesteps - 1)) * (frames.length - 1));
    const f = frames[Math.min(srcIdx, frames.length - 1)];
    out.set(f, i * featLen);
  }
  return out;
}

export const DYNAMIC_TIMESTEPS = 24;
