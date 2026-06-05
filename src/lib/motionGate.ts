import { frameVelocity } from './normalize';
import type { GateState } from './types';

export interface MotionGateConfig {
  /** Below this per-frame velocity => considered still. Tune in Train mode. */
  stillThreshold: number;
  /** Above this => considered moving. */
  moveThreshold: number;
  /** ms a hand must stay still before we emit a STATIC commit. */
  stillHoldMs: number;
  /** ms with no hand before we emit a token boundary (space). */
  pauseMs: number;
}

export const DEFAULT_GATE_CONFIG: MotionGateConfig = {
  stillThreshold: 0.012,
  moveThreshold: 0.03,
  stillHoldMs: 320,
  pauseMs: 700,
};

export type GateEvent =
  | { type: 'static'; vec: Float32Array } // hand held still long enough -> classify letter
  | { type: 'segment'; frames: Float32Array[] } // a motion segment finished -> classify word
  | { type: 'boundary' }; // hands lowered / long pause -> space

/**
 * Routes the per-frame normalized vectors to either the static classifier or
 * the dynamic classifier, solving the "is this a letter or the start of a word
 * gesture?" ambiguity. Pure state machine, no model dependency.
 *
 * Usage: call `push(vec, t)` every frame with the normalized 63-vector (or
 * null when no hand). Returns 0..1 events to act on.
 */
export class MotionGate {
  private cfg: MotionGateConfig;
  private prev: Float32Array | null = null;
  private state: GateState = 'idle';
  private stillSince = 0;
  private emittedStill = false;
  private noHandSince = 0;
  private boundaryEmitted = true; // don't emit a leading space
  private segment: Float32Array[] = [];

  constructor(cfg: MotionGateConfig = DEFAULT_GATE_CONFIG) {
    this.cfg = cfg;
  }

  setConfig(cfg: Partial<MotionGateConfig>) {
    this.cfg = { ...this.cfg, ...cfg };
  }

  /** Current router state, for UI display. */
  get currentState(): GateState {
    return this.state;
  }

  reset() {
    this.prev = null;
    this.state = 'idle';
    this.segment = [];
    this.emittedStill = false;
    this.boundaryEmitted = true;
  }

  push(vec: Float32Array | null, t: number): GateEvent[] {
    const events: GateEvent[] = [];

    if (!vec) {
      // No hand visible.
      if (this.state === 'moving' && this.segment.length > 3) {
        events.push({ type: 'segment', frames: this.segment });
        this.segment = [];
      }
      if (this.noHandSince === 0) this.noHandSince = t;
      if (!this.boundaryEmitted && t - this.noHandSince > this.cfg.pauseMs) {
        events.push({ type: 'boundary' });
        this.boundaryEmitted = true;
      }
      this.state = 'idle';
      this.prev = null;
      this.emittedStill = false;
      return events;
    }

    this.noHandSince = 0;
    this.boundaryEmitted = false;

    if (!this.prev) {
      this.prev = vec;
      this.state = 'still';
      this.stillSince = t;
      this.emittedStill = false;
      return events;
    }

    const v = frameVelocity(vec, this.prev);
    this.prev = vec;

    if (v > this.cfg.moveThreshold) {
      // Transition into / continue a motion segment.
      if (this.state !== 'moving') {
        this.state = 'moving';
        this.segment = [];
      }
      this.segment.push(vec);
      this.emittedStill = false;
    } else if (v < this.cfg.stillThreshold) {
      if (this.state === 'moving' && this.segment.length > 3) {
        // Motion just ended -> emit the completed word segment.
        events.push({ type: 'segment', frames: this.segment });
        this.segment = [];
      }
      if (this.state !== 'still') {
        this.state = 'still';
        this.stillSince = t;
        this.emittedStill = false;
      }
      // Held still long enough -> commit one static letter.
      if (!this.emittedStill && t - this.stillSince >= this.cfg.stillHoldMs) {
        events.push({ type: 'static', vec });
        this.emittedStill = true;
      }
    } else if (this.state === 'moving') {
      // In the hysteresis band while moving: keep collecting.
      this.segment.push(vec);
    }

    return events;
  }
}
