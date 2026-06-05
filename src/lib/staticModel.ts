import * as tf from '@tensorflow/tfjs';
import { FEATURES_PER_HAND, type Prediction } from './types';
import { getAllSamples, type Sample } from './dataset';

const IDB_URL = 'indexeddb://signa-static';
const BUNDLED_URL = '/models/static/model.json';
const LABELS_KEY = 'signa-static-labels';

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

/**
 * Load the static classifier. Prefers a model trained in-app (IndexedDB),
 * then a bundled model shipped in public/models/static. Returns false if no
 * model is available yet (user must train one in Train mode).
 */
export async function loadStaticModel(): Promise<boolean> {
  labels = loadLabels();
  try {
    model = await tf.loadLayersModel(IDB_URL);
    return true;
  } catch {
    /* not trained yet */
  }
  try {
    model = await tf.loadLayersModel(BUNDLED_URL);
    if (labels.length === 0) {
      // Default ASL alphabet labels for a bundled model.
      saveLabels('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''));
    }
    return true;
  } catch {
    model = null;
    return false;
  }
}

export function isStaticReady(): boolean {
  return model !== null && labels.length > 0;
}

export function staticLabels(): string[] {
  return labels;
}

/** Predict a letter from a normalized 63-vector. */
export function predictStatic(vec: Float32Array): Prediction | null {
  if (!model || labels.length === 0) return null;
  return tf.tidy(() => {
    const input = tf.tensor2d(vec, [1, FEATURES_PER_HAND]);
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

/**
 * Train a small MLP on the recorded static samples and save it to IndexedDB.
 * Returns the unique label set. Throws if there is not enough data.
 */
export async function trainStaticModel(
  epochs = 40,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const samples = await getAllSamples('static');
  return trainFromSamples(samples, epochs, onProgress);
}

async function trainFromSamples(
  samples: Sample[],
  epochs: number,
  onProgress?: (p: TrainProgress) => void
): Promise<string[]> {
  const uniqueLabels = [...new Set(samples.map((s) => s.label))].sort();
  if (uniqueLabels.length < 2) {
    throw new Error('Need at least 2 different labels to train.');
  }
  if (samples.length < uniqueLabels.length * 5) {
    throw new Error('Need at least ~5 samples per label.');
  }

  const labelIndex = new Map(uniqueLabels.map((l, i) => [l, i]));
  const xs = tf.tensor2d(
    samples.map((s) => s.data),
    [samples.length, FEATURES_PER_HAND]
  );
  const ys = tf.oneHot(
    tf.tensor1d(
      samples.map((s) => labelIndex.get(s.label)!),
      'int32'
    ),
    uniqueLabels.length
  );

  const net = tf.sequential();
  net.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [FEATURES_PER_HAND] }));
  net.add(tf.layers.dropout({ rate: 0.3 }));
  net.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  net.add(tf.layers.dense({ units: uniqueLabels.length, activation: 'softmax' }));
  net.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  await net.fit(xs, ys, {
    epochs,
    batchSize: 16,
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
