import * as tf from '@tensorflow/tfjs';
import { type Prediction } from './types';
import { getAllSamples, type Sample } from './dataset';
import { bundledPath, defaultLabels } from './languages';

// The static (letter) classifier is keyed by language pack id, so multiple
// languages can coexist. Trained models save to IndexedDB; bundled models are
// served from public/models/static/<langId>.

let model: tf.LayersModel | null = null;
let labels: string[] = [];
let activeLang = '';

const idbUrl = (lang: string) => `indexeddb://signa-${lang}-static`;
const labelsKey = (lang: string) => `signa-${lang}-static-labels`;

function loadLabels(lang: string): string[] {
  try {
    const raw = localStorage.getItem(labelsKey(lang));
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function saveLabels(lang: string, l: string[]): void {
  labels = l;
  localStorage.setItem(labelsKey(lang), JSON.stringify(l));
}

/** Bundled models ship a labels.json next to model.json (written by the trainer). */
async function fetchBundledLabels(modelUrl: string): Promise<string[] | null> {
  try {
    const res = await fetch(modelUrl.replace(/model\.json$/, 'labels.json'));
    if (!res.ok) return null;
    const data = (await res.json()) as string[];
    return Array.isArray(data) && data.length > 0 ? data : null;
  } catch {
    return null;
  }
}

/**
 * Load the static classifier for a language. Prefers an on-device trained model
 * (IndexedDB), then a bundled model. Returns false if neither exists yet.
 */
export async function loadStaticModel(lang: string): Promise<boolean> {
  activeLang = lang;
  model?.dispose();
  model = null;
  labels = loadLabels(lang);

  try {
    model = await tf.loadLayersModel(idbUrl(lang));
    return labels.length > 0;
  } catch {
    /* not trained on this device */
  }
  const bundled = bundledPath(lang, 'static');
  if (bundled) {
    try {
      model = await tf.loadLayersModel(bundled);
      const fetched = await fetchBundledLabels(bundled);
      saveLabels(lang, fetched ?? defaultLabels(lang, 'static'));
      return labels.length > 0;
    } catch {
      /* bundled file missing */
    }
  }
  model = null;
  return false;
}

export function isStaticReady(): boolean {
  return model !== null && labels.length > 0;
}

export function staticLabels(): string[] {
  return labels;
}

export function predictStatic(vec: Float32Array): Prediction | null {
  if (!model || labels.length === 0) return null;
  return tf.tidy(() => {
    const input = tf.tensor2d(vec, [1, vec.length]);
    const logits = model!.predict(input) as tf.Tensor;
    const probs = logits.dataSync();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: labels[best] ?? '?', confidence: probs[best] };
  });
}

export interface TrainProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  acc: number;
}

/** Train the static MLP on a language's recorded samples and save to IndexedDB. */
export async function trainStaticModel(
  lang: string,
  epochs = 40,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const samples = await getAllSamples('static', lang);
  return trainFromSamples(lang, samples, epochs, onProgress);
}

async function trainFromSamples(
  lang: string,
  samples: Sample[],
  epochs: number,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const uniqueLabels = [...new Set(samples.map((s) => s.label))].sort();
  if (uniqueLabels.length < 2) throw new Error('Need at least 2 different labels to train.');
  if (samples.length < uniqueLabels.length * 5) throw new Error('Need at least ~5 samples per label.');

  const featLen = samples[0].data.length; // 63 (one hand) or 126 (two hands)
  const labelIndex = new Map(uniqueLabels.map((l, i) => [l, i]));
  const xs = tf.tensor2d(samples.map((s) => s.data), [samples.length, featLen]);
  const ys = tf.oneHot(
    tf.tensor1d(samples.map((s) => labelIndex.get(s.label)!), 'int32'),
    uniqueLabels.length
  );

  const net = tf.sequential();
  net.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [featLen] }));
  net.add(tf.layers.dropout({ rate: 0.3 }));
  net.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  net.add(tf.layers.dense({ units: uniqueLabels.length, activation: 'softmax' }));
  net.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await net.fit(xs, ys, {
    epochs,
    batchSize: 16,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        onProgress?.({
          epoch: epoch + 1,
          totalEpochs: epochs,
          loss: logs?.loss ?? 0,
          acc: (logs?.acc as number) ?? (logs?.accuracy as number) ?? 0,
        }),
    },
  });

  xs.dispose();
  ys.dispose();
  model?.dispose();
  model = net;
  activeLang = lang;
  await net.save(idbUrl(lang));
  saveLabels(lang, uniqueLabels);
  return uniqueLabels;
}

export { activeLang as activeStaticLang };
