import { LANGUAGES, getLanguage } from '../lib/languages';

interface Props {
  lang: string;
  staticReady: boolean;
  dynamicReady: boolean;
  onChange: (lang: string) => void;
}

/** Dropdown to pick the active sign language / model pack. */
export default function LanguagePicker({ lang, staticReady, dynamicReady, onChange }: Props) {
  const pack = getLanguage(lang);
  const ready = staticReady || dynamicReady;

  return (
    <div className="space-y-2 rounded-xl bg-slate-900 p-3">
      <label className="text-xs uppercase tracking-wide text-slate-400">Language</label>
      <select
        value={lang}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg bg-slate-950 px-3 py-2 text-sm outline-none ring-1 ring-slate-700 focus:ring-sky-500"
      >
        {LANGUAGES.map((l) => (
          <option key={l.id} value={l.id}>
            {l.name}
          </option>
        ))}
      </select>
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">{pack.note}</span>
        {ready ? (
          <span className="rounded-full bg-emerald-700 px-2 py-0.5 text-emerald-100">
            Model ready
          </span>
        ) : (
          <span className="rounded-full bg-amber-700 px-2 py-0.5 text-amber-100">
            Train needed
          </span>
        )}
      </div>
    </div>
  );
}
