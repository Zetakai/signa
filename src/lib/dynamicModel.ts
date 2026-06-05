import * as tf from '@tensorflow/tfjs';
import { FEATURES_PER_HAND, type Prediction } from './types';
import { getAllSamples, type Sample } from './dataset';
import { DYNAMIC_TIMESTEPS } from './sequenceBuffer';
import type { TrainProgress } from './staticModel';

const IDB_URL = 'indexeddb://signa-dynamic';
const BUNDLED_URL = '/models/dynamic/model.json';
const LABELS_KEY = 'signa-dynamic-labels';

let model: tf.LayersModel | null = null;
let labels: string[] = [];

function loadLabels(): string[] {
  try {
    const raw = localStorage.getItem(LABELS_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function saveLabels(l: string[]): void {
  labels = l;
  localStorage.setItem(LABELS_KEY, JSON.stringify(l));
}

/** Load the dynamic (word) model: IndexedDB first, then bundled. */
export async function loadDynamicModel(): Promise<boolean> {
  labels = loadLabels();
  try {
    model = await tf.loadLayersModel(IDB_URL);
    return true;
  } catch {
    /* not trained yet */
  }
  try {
    model = await tf.loadLayersModel(BUNDLED_URL);
    return true;
  } catch {
    model = null;
    return false;
  }
}

export function isDynamicReady(): boolean {
  return model !== null && labels.length > 0;
}

export function dynamicLabels(): string[] {
  return labels;
}

/** Predict a word from a flattened [TIMESTEPS*63] sequence. */
export function predictDynamic(seq: Float32Array): Prediction | null {
  if (!model || labels.length === 0) return null;
  return tf.tidy(() => {
    const input = tf.tensor3d(seq, [1, DYNAMIC_TIMESTEPS, FEATURES_PER_HAND]);
    const logits = model!.predict(input) as tf.Tensor;
    const probs = logits.dataSync();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: labels[best] ?? '?', confidence: probs[best] };
  });
}

/** Train a GRU on recorded dynamic samples and save to IndexedDB. */
export async function trainDynamicModel(
  epochs = 60,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const samples = await getAllSamples('dynamic');
  return trainFromSamples(samples, epochs, onProgress);
}

async function trainFromSamples(
  samples: Sample[],
  epochs: number,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const uniqueLabels = [...new Set(samples.map((s) => s.label))].sort();
  if (uniqueLabels.length < 2) {
    throw new Error('Need at least 2 different word labels to train.');
  }
  if (samples.length < uniqueLabels.length * 5) {
    throw new Error('Need at least ~5 samples per word.');
  }

  const labelIndex = new Map(uniqueLabels.map((l, i) => [l, i]));
  const flat: number[] = [];
  for (const s of samples) flat.push(...s.data);
  const xs = tf.tensor3d(flat, [samples.length, DYNAMIC_TIMESTEPS, FEATURES_PER_HAND]);
  const ys = tf.oneHot(
    tf.tensor1d(
      samples.map((s) => labelIndex.get(s.label)!),
      'int32'
    ),
    uniqueLabels.length
  );

  const net = tf.sequential();
  net.add(
    tf.layers.gru({
      units: 64,
      inputShape: [DYNAMIC_TIMESTEPS, FEATURES_PER_HAND],
      returnSequences: false,
    })
  );
  net.add(tf.layers.dropout({ rate: 0.3 }));
  net.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  net.add(tf.layers.dense({ units: uniqueLabels.length, activation: 'softmax' }));
  net.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  await net.fit(xs, ys, {
    epochs,
    batchSize: 8,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        onProgress?.({
          epoch: epoch + 1,
          totalEpochs: epochs,
          loss: logs?.loss ?? 0,
          acc: (logs?.acc as number) ?? (logs?.accuracy as number) ?? 0,
        });
      },
    },
  });

  xs.dispose();
  ys.dispose();

  model?.dispose();
  model = net;
  await net.save(IDB_URL);
  saveLabels(uniqueLabels);
  return uniqueLabels;
}
