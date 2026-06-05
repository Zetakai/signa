import { useEffect, useState } from 'react';
import { useSignEngine } from './hooks/useSignEngine';
import CameraView from './components/CameraView';
import PredictionPanel from './components/PredictionPanel';
import Transcript from './components/Transcript';
import TrainMode from './components/TrainMode';

type Tab = 'detect' | 'train';

export default function App() {
  const { videoRef, canvasRef, state, facingRef, controls, capture } = useSignEngine();
  const [tab, setTab] = useState<Tab>('detect');

  // Pause token emission while training; resume when detecting.
  useEffect(() => {
    capture.setDetecting(tab === 'detect');
  }, [tab, capture]);

  const running = state.status === 'running';

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col gap-4 p-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-sky-300">Signa</h1>
          <p className="text-xs text-slate-400">Sign language → text, in your browser</p>
        </div>
        <div className="flex gap-1 rounded-lg bg-slate-900 p-1">
          {(['detect', 'train'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-md px-3 py-1 text-sm font-medium capitalize ${
                tab === t ? 'bg-sky-600 text-white' : 'text-slate-300'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </header>

      <CameraView videoRef={videoRef} canvasRef={canvasRef} mirror={facingRef.current === 'user'} />

      {state.status === 'idle' && (
        <button
          onClick={controls.start}
          className="rounded-lg bg-sky-600 px-4 py-3 font-medium text-white hover:bg-sky-500"
        >
          ▶ Start camera
        </button>
      )}
      {state.status === 'loading' && (
        <div className="rounded-lg bg-slate-900 p-3 text-center text-sm text-slate-400">
          Loading models…
        </div>
      )}
      {state.status === 'error' && (
        <div className="rounded-lg bg-red-950 p-3 text-sm text-red-300">{state.error}</div>
      )}

      {running && (
        <div className="flex gap-2">
          {state.canSwitchCamera && (
            <button
              onClick={controls.switchCamera}
              className="rounded-lg bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700"
            >
              ⟲ Flip camera
            </button>
          )}
          <button
            onClick={controls.stop}
            className="rounded-lg bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700"
          >
            ⏹ Stop
          </button>
        </div>
      )}

      {running && tab === 'detect' && (
        <>
          <PredictionPanel
            prediction={state.prediction}
            gateState={state.gateState}
            fps={state.fps}
          />
          {!state.staticReady && !state.dynamicReady && (
            <div className="rounded-lg bg-amber-950 p-3 text-sm text-amber-300">
              No trained model yet. Go to <b>Train</b> to teach Signa some signs, then come back.
            </div>
          )}
          <Transcript
            transcript={state.transcript}
            onClear={controls.clear}
            onBackspace={controls.backspace}
          />
        </>
      )}

      {running && tab === 'train' && (
        <TrainMode
          getLatestVec={capture.getLatestVec}
          setSeqSink={capture.setSeqSink}
          onTrained={controls.refreshModels}
        />
      )}

      {!running && (
        <p className="text-center text-xs text-slate-500">
          Everything runs locally. Your camera never leaves the device.
        </p>
      )}
    </div>
  );
}
