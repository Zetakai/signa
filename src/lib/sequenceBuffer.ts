import { FEATURES_PER_HAND } from './types';

/**
 * Resample a variable-length motion segment to a fixed number of timesteps so
 * the dynamic model always sees the same input shape [TIMESTEPS, 63].
 * Uses nearest-neighbour resampling along the time axis.
 */
export function resampleSequence(frames: Float32Array[], timesteps: number): Float32Array {
  const out = new Float32Array(timesteps * FEATURES_PER_HAND);
  if (frames.length === 0) return out;

  for (let i = 0; i < timesteps; i++) {
    const srcIdx =
      frames.length === 1 ? 0 : Math.round((i / (timesteps - 1)) * (frames.length - 1));
    const f = frames[Math.min(srcIdx, frames.length - 1)];
    out.set(f, i * FEATURES_PER_HAND);
  }
  return out;
}

export const DYNAMIC_TIMESTEPS = 24;
