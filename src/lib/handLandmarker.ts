import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision';
import type { FrameResult, Hand } from './types';

// Self-hosted assets (see public/models). Avoids a third-party CDN runtime
// dependency so the app is reliable + offline-capable on Cloudflare Pages.
const WASM_PATH = '/models/wasm';
const TASK_PATH = '/models/hand_landmarker.task';

let landmarker: HandLandmarker | null = null;
let lastVideoTime = -1;

/** Initialize the MediaPipe HandLandmarker once (VIDEO mode). */
export async function initHandLandmarker(numHands = 1): Promise<void> {
  if (landmarker) return;
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: TASK_PATH,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands,
  });
}

export function isLandmarkerReady(): boolean {
  return landmarker !== null;
}

/**
 * Detect hands in the current video frame. Returns null if the video frame
 * hasn't advanced (avoids re-running on the same frame).
 */
export function detect(video: HTMLVideoElement, timestampMs: number): FrameResult | null {
  if (!landmarker) return null;
  if (video.currentTime === lastVideoTime) return null;
  lastVideoTime = video.currentTime;

  const res: HandLandmarkerResult = landmarker.detectForVideo(video, timestampMs);
  const hands: Hand[] = (res.landmarks ?? []).map((lm) =>
    lm.map((p) => ({ x: p.x, y: p.y, z: p.z }))
  );
  return { hands, timestampMs };
}

export function disposeHandLandmarker(): void {
  landmarker?.close();
  landmarker = null;
  lastVideoTime = -1;
}
