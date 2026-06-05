import { useEffect, useRef, useState } from 'react';
import {
  addSample,
  getAllSamples,
  clearSamples,
  countByLabel,
  exportSamples,
} from '../lib/dataset';
import { trainStaticModel, type TrainProgress } from '../lib/staticModel';
import { trainDynamicModel } from '../lib/dynamicModel';
import { resampleSequence, DYNAMIC_TIMESTEPS } from '../lib/sequenceBuffer';

type Kind = 'static' | 'dynamic';

interface Props {
  lang: string;
  langName: string;
  getLatestVec: () => Float32Array | null;
  setSeqSink: (sink: ((vec: Float32Array | null) => void) | null) => void;
  onTrained: () => void;
}

const STATIC_BURST = 25; // frames captured per static sample press

export default function TrainMode({ lang, langName, getLatestVec, setSeqSink, onTrained }: Props) {
  const [kind, setKind] = useState<Kind>('static');
  const [label, setLabel] = useState('');
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [recording, setRecording] = useState(false);
  const recBuffer = useRef<Float32Array[]>([]);

  const refreshCounts = async (k: Kind) => {
    setCounts(countByLabel(await getAllSamples(k, lang)));
  };

  useEffect(() => {
    refreshCounts(kind);
    setProgress(null);
  }, [kind, lang]);

  // --- Static capture: burst of frames ---
  const captureStatic = async () => {
    const l = label.trim().toUpperCase();
    if (!l) return setStatus('Enter a label first.');
    setBusy(true);
    setStatus(`Capturing "${l}"… hold the pose`);
    let n = 0;
    await new Promise<void>((resolve) => {
      const id = setInterval(async () => {
        const vec = getLatestVec();
        if (vec && vec.some((v) => v !== 0)) {
          await addSample('static', { lang, label: l, data: Array.from(vec) });
          n++;
        }
        if (n >= STATIC_BURST) {
          clearInterval(id);
          resolve();
        }
      }, 55);
    });
    setStatus(`Saved ${n} samples for "${l}".`);
    await refreshCounts('static');
    setBusy(false);
  };

  // --- Dynamic capture: record a motion segment ---
  const toggleRecord = () => {
    const l = label.trim().toUpperCase();
    if (!recording) {
      if (!l) return setStatus('Enter a word label first.');
      recBuffer.current = [];
      setSeqSink((vec) => {
        if (vec && vec.some((v) => v !== 0)) recBuffer.current.push(vec);
      });
      setRecording(true);
      setStatus(`Recording "${l}"… perform the gesture, then Stop.`);
    } else {
      setSeqSink(null);
      setRecording(false);
      const frames = recBuffer.current;
      if (frames.length < 4) {
        setStatus('Too short — try again.');
        return;
      }
      const seq = resampleSequence(frames, DYNAMIC_TIMESTEPS);
      addSample('dynamic', { lang, label: l, data: Array.from(seq) }).then(() => {
        setStatus(`Saved 1 sequence for "${l}" (${frames.length} frames).`);
        refreshCounts('dynamic');
      });
    }
  };

  useEffect(() => () => setSeqSink(null), [setSeqSink]);

  const train = async () => {
    setBusy(true);
    setProgress(null);
    setStatus('Training…');
    try {
      const onProg = (p: TrainProgress) => setProgress(p);
      const labels =
        kind === 'static'
          ? await trainStaticModel(lang, 40, onProg)
          : await trainDynamicModel(lang, 60, onProg);
      setStatus(`Trained on ${labels.length} labels: ${labels.join(', ')}`);
      onTrained();
    } catch (err) {
      setStatus((err as Error).message);
    }
    setBusy(false);
  };

  const onClear = async () => {
    await clearSamples(kind, lang);
    await refreshCounts(kind);
    setStatus('Cleared samples.');
  };

  const onExport = async () => {
    exportSamples(kind, await getAllSamples(kind, lang));
  };

  const labelList = Object.entries(counts).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div className="space-y-4 rounded-xl bg-slate-900 p-4">
      <div className="rounded-lg bg-slate-950 px-3 py-2 text-xs text-slate-400">
        Training: <span className="font-medium text-sky-300">{langName}</span>
      </div>
      <div className="flex gap-2">
        {(['static', 'dynamic'] as Kind[]).map((k) => (
          <button
            key={k}
            onClick={() => setKind(k)}
            className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium ${
              kind === k ? 'bg-sky-600 text-white' : 'bg-slate-800 text-slate-300'
            }`}
          >
            {k === 'static' ? 'Letters (static)' : 'Words (dynamic)'}
          </button>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder={kind === 'static' ? 'Letter e.g. A' : 'Word e.g. HELLO'}
          maxLength={kind === 'static' ? 2 : 20}
          className="flex-1 rounded-lg bg-slate-950 px-3 py-2 text-sm outline-none ring-1 ring-slate-700 focus:ring-sky-500"
        />
        {kind === 'static' ? (
          <button
            onClick={captureStatic}
            disabled={busy}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            Capture
          </button>
        ) : (
          <button
            onClick={toggleRecord}
            disabled={busy}
            className={`rounded-lg px-4 py-2 text-sm font-medium text-white disabled:opacity-40 ${
              recording ? 'bg-red-600 hover:bg-red-500' : 'bg-emerald-600 hover:bg-emerald-500'
            }`}
          >
            {recording ? 'Stop' : 'Record'}
          </button>
        )}
      </div>

      {status && <div className="text-xs text-slate-400">{status}</div>}

      {progress && (
        <div className="text-xs text-sky-400">
          Epoch {progress.epoch}/{progress.totalEpochs} — loss{' '}
          {progress.loss.toFixed(3)} — acc {(progress.acc * 100).toFixed(0)}%
        </div>
      )}

      {labelList.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {labelList.map(([l, c]) => (
            <span key={l} className="rounded-full bg-slate-800 px-2 py-1 text-xs">
              {l}: {c}
            </span>
          ))}
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          onClick={train}
          disabled={busy || labelList.length < 2}
          className="flex-1 rounded-lg bg-sky-600 px-3 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
        >
          Train model
        </button>
        <button onClick={onExport} className="rounded-lg bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700">
          Export
        </button>
        <button onClick={onClear} className="rounded-lg bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700">
          Clear
        </button>
      </div>

      <p className="text-xs text-slate-500">
        Tip: capture {kind === 'static' ? '≥5 bursts per letter from slightly different angles' : '≥5 recordings per word'} for
        usable accuracy. Models save to your browser (IndexedDB) — they stay on this device.
      </p>
    </div>
  );
}
