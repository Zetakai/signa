// Shared types for the landmark pipeline.

/** A single 3D hand keypoint from MediaPipe (normalized 0..1 image coords, z relative). */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** One hand = 21 landmarks. */
export type Hand = Landmark[];

/** Result of one detection frame. */
export interface FrameResult {
  hands: Hand[]; // 0..numHands
  timestampMs: number;
}

/** A prediction with class label + confidence 0..1. */
export interface Prediction {
  label: string;
  confidence: number;
}

/** Tokens emitted into the sentence builder. */
export type Token =
  | { kind: 'letter'; value: string; confidence: number }
  | { kind: 'word'; value: string; confidence: number }
  | { kind: 'boundary' }; // space / end-of-word

/** Mode of the motion-gate router. */
export type GateState = 'idle' | 'still' | 'moving';

export const NUM_LANDMARKS = 21;
export const FEATURES_PER_HAND = NUM_LANDMARKS * 3; // 63
