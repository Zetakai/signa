import type { GateState, Prediction } from '../lib/types';

interface Props {
  prediction: Prediction | null;
  gateState: GateState;
  fps: number;
}

const GATE_LABEL: Record<GateState, string> = {
  idle: 'No hand',
  still: 'Still → letter',
  moving: 'Moving → word',
};

const GATE_COLOR: Record<GateState, string> = {
  idle: 'bg-slate-600',
  still: 'bg-emerald-600',
  moving: 'bg-amber-600',
};

/** Shows the latest prediction, the router state, and FPS. */
export default function PredictionPanel({ prediction, gateState, fps }: Props) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-900 p-3">
      <div className="flex items-center gap-3">
        <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-slate-800 text-2xl font-bold text-sky-300">
          {prediction?.label ?? '–'}
        </div>
        <div className="text-sm">
          <div className="text-slate-400">Last sign</div>
          <div className="font-medium">
            {prediction ? `${Math.round(prediction.confidence * 100)}% confident` : 'waiting…'}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs">
        <span className={`rounded-full px-2 py-1 text-white ${GATE_COLOR[gateState]}`}>
          {GATE_LABEL[gateState]}
        </span>
        <span className="text-slate-500">{fps} fps</span>
      </div>
    </div>
  );
}
