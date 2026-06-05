import * as tf from '@tensorflow/tfjs';
import { type Prediction } from './types';
import { getAllSamples, type Sample } from './dataset';
import { DYNAMIC_TIMESTEPS } from './sequenceBuffer';
import { bundledPath, defaultLabels } from './languages';
import type { TrainProgress } from './staticModel';

// Dynamic (word) classifier, keyed by language pack id.

let model: tf.LayersModel | null = null;
let labels: string[] = [];

const idbUrl = (lang: string) => `indexeddb://signa-${lang}-dynamic`;
const labelsKey = (lang: string) => `signa-${lang}-dynamic-labels`;

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

export async function loadDynamicModel(lang: string): Promise<boolean> {
  model?.dispose();
  model = null;
  labels = loadLabels(lang);

  try {
    model = await tf.loadLayersModel(idbUrl(lang));
    return labels.length > 0;
  } catch {
    /* not trained */
  }
  const bundled = bundledPath(lang, 'dynamic');
  if (bundled) {
    try {
      model = await tf.loadLayersModel(bundled);
      const fetched = await fetchBundledLabels(bundled);
      saveLabels(lang, fetched ?? defaultLabels(lang, 'dynamic'));
      return labels.length > 0;
    } catch {
      /* missing */
    }
  }
  model = null;
  return false;
}

export function isDynamicReady(): boolean {
  return model !== null && labels.length > 0;
}

export function dynamicLabels(): string[] {
  return labels;
}

export function predictDynamic(seq: Float32Array): Prediction | null {
  if (!model || labels.length === 0) return null;
  return tf.tidy(() => {
    const featLen = seq.length / DYNAMIC_TIMESTEPS;
    const input = tf.tensor3d(seq, [1, DYNAMIC_TIMESTEPS, featLen]);
    const logits = model!.predict(input) as tf.Tensor;
    const probs = logits.dataSync();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: labels[best] ?? '?', confidence: probs[best] };
  });
}

export async function trainDynamicModel(
  lang: string,
  epochs = 60,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const samples = await getAllSamples('dynamic', lang);
  return trainFromSamples(lang, samples, epochs, onProgress);
}

async function trainFromSamples(
  lang: string,
  samples: Sample[],
  epochs: number,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const uniqueLabels = [...new Set(samples.map((s) => s.label))].sort();
  if (uniqueLabels.length < 2) throw new Error('Need at least 2 different word labels to train.');
  if (samples.length < uniqueLabels.length * 5) throw new Error('Need at least ~5 samples per word.');

  const featLen = samples[0].data.length / DYNAMIC_TIMESTEPS;
  const labelIndex = new Map(uniqueLabels.map((l, i) => [l, i]));
  const flat: number[] = [];
  for (const s of samples) flat.push(...s.data);
  const xs = tf.tensor3d(flat, [samples.length, DYNAMIC_TIMESTEPS, featLen]);
  const ys = tf.oneHot(
    tf.tensor1d(samples.map((s) => labelIndex.get(s.label)!), 'int32'),
    uniqueLabels.length
  );

  const net = tf.sequential();
  net.add(tf.layers.gru({ units: 64, inputShape: [DYNAMIC_TIMESTEPS, featLen], returnSequences: false }));
  net.add(tf.layers.dropout({ rate: 0.3 }));
  net.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  net.add(tf.layers.dense({ units: uniqueLabels.length, activation: 'softmax' }));
  net.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  await net.fit(xs, ys, {
    epochs,
    batchSize: 8,
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
  await net.save(idbUrl(lang));
  saveLabels(lang, uniqueLabels);
  return uniqueLabels;
}
