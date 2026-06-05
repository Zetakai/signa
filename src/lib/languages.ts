// Registry of sign languages / model packs shown in the picker.
//
// Each language has an optional `static` (fingerspelled letters) and `dynamic`
// (word gestures) model. A model is "bundled" when shipped under public/models
// (set `bundled: true` once weights are committed). If not bundled and not
// trained on-device yet, the picker routes the user to the Train tab.

export type ModelType = 'static' | 'dynamic';

export interface ModelSlot {
  /** True when weights are shipped in public/models/<type>/<langId>/model.json. */
  bundled: boolean;
  /** Default labels for a bundled model (optional; trained models store their own). */
  labels?: string[];
}

export interface LanguagePack {
  id: string;
  name: string;
  /** Short note shown under the name. */
  note?: string;
  /** Hands this language uses: 1 (most fingerspelling) or 2 (e.g. BISINDO). */
  hands: 1 | 2;
  static?: ModelSlot;
  dynamic?: ModelSlot;
}

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

export const LANGUAGES: LanguagePack[] = [
  {
    id: 'asl',
    name: 'American Sign Language',
    note: 'Letters (A–Z) ready • words: train',
    hands: 1,
    static: { bundled: true, labels: ALPHABET },
    dynamic: { bundled: false },
  },
  {
    id: 'sibi',
    name: 'SIBI (Bahasa Isyarat Indonesia)',
    note: 'Letters (A–Y, no J/Z) ready • words: train',
    hands: 1,
    static: { bundled: true, labels: ALPHABET.filter((l) => l !== 'J' && l !== 'Z') },
    dynamic: { bundled: false },
  },
  {
    id: 'bisindo',
    name: 'BISINDO (Indonesia, two-handed)',
    note: 'Letters (A–Z) ready • two-handed • words: train',
    hands: 2,
    static: { bundled: true, labels: ALPHABET },
    dynamic: { bundled: false },
  },
  {
    id: 'custom',
    name: 'Custom',
    note: 'Train your own signs',
    hands: 1,
    static: { bundled: false },
    dynamic: { bundled: false },
  },
];

export function getLanguage(id: string): LanguagePack {
  return LANGUAGES.find((l) => l.id === id) ?? LANGUAGES[0];
}

/** Path to a bundled model.json for a language/type, or null if not bundled. */
export function bundledPath(langId: string, type: ModelType): string | null {
  const pack = getLanguage(langId);
  const slot = type === 'static' ? pack.static : pack.dynamic;
  return slot?.bundled ? `/models/${type}/${langId}/model.json` : null;
}

export function defaultLabels(langId: string, type: ModelType): string[] {
  const pack = getLanguage(langId);
  const slot = type === 'static' ? pack.static : pack.dynamic;
  return slot?.labels ?? [];
}
