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


def extract_dataset(dataset_dir, limit=300, only_letters=True):
    import cv2
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    candidates = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    # Skip non-letter folders like del/nothing/space when only_letters.
    labels = [d for d in candidates if (not only_letters or (len(d) == 1 and d.isalpha()))]
    print(f"Labels: {labels}  (limit {limit}/label)")

    X, y = [], []
    kept_labels = []
    for label in labels:
        folder = os.path.join(dataset_dir, label)
        files = sorted(os.listdir(folder))[:limit]
        count = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            X.append(normalize_hand(res.multi_hand_landmarks[0].landmark))
            y.append(label)
            count += 1
        print(f"  {label}: {count} detected")
        if count >= 5:
            kept_labels.append(label)

    hands.close()
    return np.array(X, dtype=np.float32), y, kept_labels


def train(X, y, labels, epochs):
    import tensorflow as tf

    label_index = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_index[v] for v in y], dtype=np.int32)
    Y = tf.keras.utils.to_categorical(y_idx, num_classes=len(labels))

    # Shuffle BEFORE the validation split. Keras' validation_split carves off
    # the tail of the array without shuffling first; the data is grouped by
    # label, so an unshuffled split validates on only the last few classes.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]

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
    ap.add_argument("--limit", type=int, default=300, help="max images per label")
    ap.add_argument("--all-folders", action="store_true",
                    help="include non-letter folders (del/nothing/space)")
    args = ap.parse_args()

    cache = os.path.join("training", f".cache_{args.lang}_{args.limit}.npz")
    if os.path.exists(cache):
        print(f"Loading cached landmarks from {cache}")
        d = np.load(cache, allow_pickle=True)
        X, y, labels = d["X"], list(d["y"]), list(d["labels"])
    else:
        X, y, labels = extract_dataset(args.dataset, args.limit, not args.all_folders)
        np.savez(cache, X=X, y=np.array(y), labels=np.array(labels))
        print(f"Cached landmarks to {cache}")

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
