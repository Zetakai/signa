import { useCallback, useEffect, useRef, useState } from 'react';
import { startCamera, hasMultipleCameras, type CameraHandle, type Facing } from '../lib/camera';
import { initHandLandmarker, detect, disposeHandLandmarker } from '../lib/handLandmarker';
import { normalizeHand } from '../lib/normalize';
import { drawHands } from '../lib/drawLandmarks';
import { MotionGate } from '../lib/motionGate';
import { resampleSequence, DYNAMIC_TIMESTEPS } from '../lib/sequenceBuffer';
import { SentenceBuilder } from '../lib/sentenceBuilder';
import {
  loadStaticModel,
  predictStatic,
  isStaticReady,
} from '../lib/staticModel';
import {
  loadDynamicModel,
  predictDynamic,
  isDynamicReady,
} from '../lib/dynamicModel';
import type { GateState, Prediction } from '../lib/types';
import { LANGUAGES } from '../lib/languages';

export type EngineStatus = 'idle' | 'loading' | 'running' | 'error';

// Confidence floors below which a prediction is ignored.
const STATIC_MIN_CONF = 0.6;
const DYNAMIC_MIN_CONF = 0.6;

export interface EngineState {
  status: EngineStatus;
  error: string | null;
  gateState: GateState;
  prediction: Prediction | null;
  transcript: string;
  fps: number;
  lang: string;
  staticReady: boolean;
  dynamicReady: boolean;
  canSwitchCamera: boolean;
}

export function useSignEngine() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const cameraRef = useRef<CameraHandle | null>(null);
  const gateRef = useRef(new MotionGate());
  const builderRef = useRef(new SentenceBuilder());
  const rafRef = useRef<number | null>(null);
  const facingRef = useRef<Facing>('user');
  const langRef = useRef<string>(LANGUAGES[0].id);

  // Detection on/off (Train mode pauses token emission but keeps the loop).
  const detectingRef = useRef(true);
  // Latest normalized vector + a live capture sink for Train mode.
  const latestVecRef = useRef<Float32Array | null>(null);
  const seqSinkRef = useRef<((vec: Float32Array | null) => void) | null>(null);

  // FPS bookkeeping.
  const fpsRef = useRef({ last: performance.now(), frames: 0 });

  const [state, setState] = useState<EngineState>({
    status: 'idle',
    error: null,
    gateState: 'idle',
    prediction: null,
    transcript: '',
    fps: 0,
    lang: LANGUAGES[0].id,
    staticReady: false,
    dynamicReady: false,
    canSwitchCamera: false,
  });

  const patch = useCallback((p: Partial<EngineState>) => {
    setState((s) => ({ ...s, ...p }));
  }, []);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }

    const now = performance.now();
    const result = detect(video, now);

    if (result) {
      // Size canvas to the displayed video.
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext('2d');
      if (ctx) {
        drawHands(ctx, result.hands, canvas.width, canvas.height, facingRef.current === 'user');
      }

      const hand = result.hands[0];
      const vec = hand ? normalizeHand(hand) : null;
      latestVecRef.current = vec;
      seqSinkRef.current?.(vec);

      if (detectingRef.current) {
        const events = gateRef.current.push(vec, now);
        for (const ev of events) {
          if (ev.type === 'static') {
            const pred = predictStatic(ev.vec);
            if (pred && pred.confidence >= STATIC_MIN_CONF) {
              builderRef.current.push({
                kind: 'letter',
                value: pred.label,
                confidence: pred.confidence,
              });
              patch({ prediction: pred, transcript: builderRef.current.transcript });
            }
          } else if (ev.type === 'segment') {
            const seq = resampleSequence(ev.frames, DYNAMIC_TIMESTEPS);
            const pred = predictDynamic(seq);
            if (pred && pred.confidence >= DYNAMIC_MIN_CONF) {
              builderRef.current.push({
                kind: 'word',
                value: pred.label,
                confidence: pred.confidence,
              });
              patch({ prediction: pred, transcript: builderRef.current.transcript });
            }
          } else if (ev.type === 'boundary') {
            builderRef.current.push({ kind: 'boundary' });
            patch({ transcript: builderRef.current.transcript });
          }
        }
      }
    }

    // FPS update ~ once/sec.
    const f = fpsRef.current;
    f.frames++;
    if (now - f.last >= 1000) {
      patch({ fps: Math.round((f.frames * 1000) / (now - f.last)), gateState: gateRef.current.currentState });
      f.frames = 0;
      f.last = now;
    }

    rafRef.current = requestAnimationFrame(loop);
  }, [patch]);

  const start = useCallback(async () => {
    if (cameraRef.current) return;
    patch({ status: 'loading', error: null });
    try {
      await initHandLandmarker(1);
      const sReady = await loadStaticModel(langRef.current);
      const dReady = await loadDynamicModel(langRef.current);
      const multi = await hasMultipleCameras();

      const video = videoRef.current!;
      cameraRef.current = await startCamera(video, facingRef.current);

      patch({
        status: 'running',
        staticReady: sReady && isStaticReady(),
        dynamicReady: dReady && isDynamicReady(),
        canSwitchCamera: multi,
      });
      gateRef.current.reset();
      rafRef.current = requestAnimationFrame(loop);
    } catch (err) {
      patch({ status: 'error', error: (err as Error).message });
    }
  }, [loop, patch]);

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    cameraRef.current?.stop();
    cameraRef.current = null;
    patch({ status: 'idle' });
  }, [patch]);

  const switchCamera = useCallback(async () => {
    facingRef.current = facingRef.current === 'user' ? 'environment' : 'user';
    cameraRef.current?.stop();
    cameraRef.current = null;
    try {
      const video = videoRef.current!;
      cameraRef.current = await startCamera(video, facingRef.current);
    } catch (err) {
      patch({ status: 'error', error: (err as Error).message });
    }
  }, [patch]);

  const clear = useCallback(() => {
    builderRef.current.clear();
    gateRef.current.reset();
    patch({ transcript: '', prediction: null });
  }, [patch]);

  const backspace = useCallback(() => {
    builderRef.current.backspace();
    patch({ transcript: builderRef.current.transcript });
  }, [patch]);

  /** Refresh model-ready flags after the user trains a new model. */
  const refreshModels = useCallback(async () => {
    const s = await loadStaticModel(langRef.current);
    const d = await loadDynamicModel(langRef.current);
    patch({ staticReady: s && isStaticReady(), dynamicReady: d && isDynamicReady() });
  }, [patch]);

  /** Switch the active language pack and (re)load its models live. */
  const setLanguage = useCallback(
    async (lang: string) => {
      langRef.current = lang;
      builderRef.current.clear();
      gateRef.current.reset();
      patch({ lang, transcript: '', prediction: null });
      const s = await loadStaticModel(lang);
      const d = await loadDynamicModel(lang);
      patch({ staticReady: s && isStaticReady(), dynamicReady: d && isDynamicReady() });
    },
    [patch]
  );

  // --- Train-mode capture API ---
  const setDetecting = useCallback((on: boolean) => {
    detectingRef.current = on;
    if (on) gateRef.current.reset();
  }, []);

  const getLatestVec = useCallback(() => latestVecRef.current, []);

  const setSeqSink = useCallback((sink: ((vec: Float32Array | null) => void) | null) => {
    seqSinkRef.current = sink;
  }, []);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      cameraRef.current?.stop();
      disposeHandLandmarker();
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    state,
    facingRef,
    langRef,
    controls: { start, stop, switchCamera, clear, backspace, refreshModels, setLanguage },
    capture: { setDetecting, getLatestVec, setSeqSink },
  };
}
