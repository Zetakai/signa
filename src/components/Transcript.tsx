import { useState } from 'react';
import { smoothTranscript } from '../lib/smooth';

interface Props {
  transcript: string;
  onClear: () => void;
  onBackspace: () => void;
}

/** Running transcript with raw gloss + optional LLM-smoothed English. */
export default function Transcript({ transcript, onClear, onBackspace }: Props) {
  const [smoothed, setSmoothed] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [offlineNote, setOfflineNote] = useState(false);

  const onSmooth = async () => {
    if (!transcript.trim()) return;
    setLoading(true);
    setOfflineNote(false);
    const res = await smoothTranscript(transcript);
    setSmoothed(res.text);
    setOfflineNote(res.source === 'offline-fallback');
    setLoading(false);
  };

  return (
    <div className="space-y-3 rounded-xl bg-slate-900 p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-slate-400">Transcript (gloss)</span>
        <div className="flex gap-2">
          <button
            onClick={onBackspace}
            className="rounded-md bg-slate-800 px-2 py-1 text-xs hover:bg-slate-700"
          >
            ⌫ Back
          </button>
          <button
            onClick={() => {
              onClear();
              setSmoothed(null);
            }}
            className="rounded-md bg-slate-800 px-2 py-1 text-xs hover:bg-slate-700"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="min-h-[3rem] rounded-lg bg-slate-950 p-3 text-lg font-medium">
        {transcript || <span className="text-slate-600">Start signing…</span>}
      </div>

      <button
        onClick={onSmooth}
        disabled={loading || !transcript.trim()}
        className="w-full rounded-lg bg-sky-600 px-3 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
      >
        {loading ? 'Smoothing…' : '✨ Smooth to English'}
      </button>

      {smoothed !== null && (
        <div className="rounded-lg border border-sky-900 bg-slate-950 p-3">
          <div className="text-xs uppercase tracking-wide text-sky-400">English</div>
          <div className="mt-1 text-lg">{smoothed}</div>
          {offlineNote && (
            <div className="mt-1 text-xs text-amber-500">
              Offline or smoothing unavailable — showing raw gloss.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
