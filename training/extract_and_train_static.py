#!/usr/bin/env python3
"""
Train a STATIC (fingerspelled letter) sign model from an image dataset and
export it to TF.js so the web app can load it as a bundled model.

Dataset layout (one folder per label, images inside):

    dataset/
        A/ img1.jpg img2.jpg ...
        B/ ...
        ...

The script:
  1. runs MediaPipe Hands on each image to get 21 keypoints,
  2. normalizes them the SAME way as src/lib/normalize.ts (wrist-relative +
     scale), producing a 63-vector,
  3. trains an MLP identical to src/lib/staticModel.ts,
  4. exports model.json + weights + labels.json into
     public/models/static/<lang>/.

Then set `static.bundled = true` for that language in src/lib/languages.ts and
commit the produced files.

Usage:
    python training/extract_and_train_static.py \
        --dataset ./dataset --lang asl --epochs 40
"""
import argparse
import json
import os

import numpy as np

NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3  # 63


def normalize_hand(landmarks):
    """Match src/lib/normalize.ts: wrist origin + scale by max distance."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0]
    rel = pts - wrist
    dists = np.linalg.norm(rel, axis=1)
    scale = max(float(dists.max()), 1e-6)
    return (rel / scale).reshape(-1)  # (63,)


def extract_dataset(dataset_dir):
    import cv2
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    X, y = [], []
    labels = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    print(f"Labels: {labels}")

    for label in labels:
        folder = os.path.join(dataset_dir, label)
        count = 0
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            vec = normalize_hand(res.multi_hand_landmarks[0].landmark)
            X.append(vec)
            y.append(label)
            count += 1
        print(f"  {label}: {count} samples")

    hands.close()
    return np.array(X, dtype=np.float32), y, labels


def train(X, y, labels, epochs):
    import tensorflow as tf

    label_index = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_index[v] for v in y], dtype=np.int32)
    Y = tf.keras.utils.to_categorical(y_idx, num_classes=len(labels))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURES,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, Y, epochs=epochs, batch_size=16, shuffle=True,
              validation_split=0.15)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="root folder, one subdir per label")
    ap.add_argument("--lang", required=True, help="language pack id, e.g. asl / sibi")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    X, y, labels = extract_dataset(args.dataset)
    if len(X) < len(labels) * 5:
        raise SystemExit("Not enough detected samples (need ~5+ per label).")

    model = train(X, y, labels, args.epochs)

    import tensorflowjs as tfjs
    out_dir = os.path.join("public", "models", "static", args.lang)
    os.makedirs(out_dir, exist_ok=True)
    tfjs.converters.save_keras_model(model, out_dir)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"\nExported to {out_dir}")
    print(f"Now set static.bundled = true for '{args.lang}' in src/lib/languages.ts")


if __name__ == "__main__":
    main()
