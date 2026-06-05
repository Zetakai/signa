// Minimal IndexedDB store for recorded training samples.
// Static samples: a 63-vector + label. Dynamic samples: a flattened
// [TIMESTEPS*63] vector + label.

const DB_NAME = 'signa';
const DB_VERSION = 1;
const STORE_STATIC = 'static-samples';
const STORE_DYNAMIC = 'dynamic-samples';

export interface Sample {
  id?: number;
  lang: string; // language pack id this sample belongs to
  label: string;
  data: number[]; // flattened features
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_STATIC)) {
        db.createObjectStore(STORE_STATIC, { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains(STORE_DYNAMIC)) {
        db.createObjectStore(STORE_DYNAMIC, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function storeName(kind: 'static' | 'dynamic'): string {
  return kind === 'static' ? STORE_STATIC : STORE_DYNAMIC;
}

export async function addSample(kind: 'static' | 'dynamic', sample: Sample): Promise<void> {
  const db = await openDB();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(storeName(kind), 'readwrite');
    tx.objectStore(storeName(kind)).add(sample);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

/** Get samples for a kind, optionally filtered to one language pack. */
export async function getAllSamples(
  kind: 'static' | 'dynamic',
  lang?: string
): Promise<Sample[]> {
  const db = await openDB();
  const samples = await new Promise<Sample[]>((resolve, reject) => {
    const tx = db.transaction(storeName(kind), 'readonly');
    const req = tx.objectStore(storeName(kind)).getAll();
    req.onsuccess = () => resolve(req.result as Sample[]);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return lang ? samples.filter((s) => s.lang === lang) : samples;
}

/** Clear samples for a kind. If `lang` given, only that language's samples. */
export async function clearSamples(kind: 'static' | 'dynamic', lang?: string): Promise<void> {
  if (lang) {
    const remaining = (await getAllSamples(kind)).filter((s) => s.lang !== lang);
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(storeName(kind), 'readwrite');
      const store = tx.objectStore(storeName(kind));
      store.clear();
      for (const s of remaining) store.add({ lang: s.lang, label: s.label, data: s.data });
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
    return;
  }
  const db = await openDB();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(storeName(kind), 'readwrite');
    tx.objectStore(storeName(kind)).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

/** Count samples per label, for UI. */
export function countByLabel(samples: Sample[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of samples) out[s.label] = (out[s.label] ?? 0) + 1;
  return out;
}

/** Export samples as a downloadable JSON blob. */
export function exportSamples(kind: 'static' | 'dynamic', samples: Sample[]): void {
  const blob = new Blob([JSON.stringify({ kind, samples }, null, 0)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `signa-${kind}-samples.json`;
  a.click();
  URL.revokeObjectURL(url);
}
