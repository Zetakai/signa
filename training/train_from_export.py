#!/usr/bin/env python3
"""
Train a model from data exported by the app's Train tab (the "Export" button),
and export it to TF.js as a bundled model. Works for both static (letters) and
dynamic (words) data — the kind is read from the JSON.

This lets you collect data in-browser on real devices, then bake the result
into a shippable model everyone gets by default.

Usage:
    python training/train_from_export.py --input signa-static-samples.json --lang asl
    python training/train_from_export.py --input signa-dynamic-samples.json --lang asl --epochs 60
"""
import argparse
import json
import os

import numpy as np

FEATURES = 63
DYNAMIC_TIMESTEPS = 24  # must match src/lib/sequenceBuffer.ts


def build_static(n_classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURES,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def build_dynamic(n_classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(DYNAMIC_TIMESTEPS, FEATURES)),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def main():
    import tensorflow as tf
    import tensorflowjs as tfjs

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="exported signa JSON file")
    ap.add_argument("--lang", required=True, help="language pack id, e.g. asl")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        blob = json.load(f)
    kind = blob["kind"]  # 'static' | 'dynamic'
    samples = blob["samples"]
    labels = sorted({s["label"] for s in samples})
    label_index = {l: i for i, l in enumerate(labels)}
    print(f"kind={kind} labels={labels} n={len(samples)}")

    data = np.array([s["data"] for s in samples], dtype=np.float32)
    y_idx = np.array([label_index[s["label"]] for s in samples], dtype=np.int32)
    Y = tf.keras.utils.to_categorical(y_idx, num_classes=len(labels))

    # Shuffle before fit so Keras' tail validation_split isn't all one label.
    perm = np.random.default_rng(42).permutation(len(data))
    data, Y = data[perm], Y[perm]

    if kind == "static":
        X = data.reshape(-1, FEATURES)
        model = build_static(len(labels))
        epochs = args.epochs or 40
        batch = 16
    else:
        X = data.reshape(-1, DYNAMIC_TIMESTEPS, FEATURES)
        model = build_dynamic(len(labels))
        epochs = args.epochs or 60
        batch = 8

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, Y, epochs=epochs, batch_size=batch, shuffle=True,
              validation_split=0.15)

    out_dir = os.path.join("public", "models", kind, args.lang)
    os.makedirs(out_dir, exist_ok=True)
    tfjs.converters.save_keras_model(model, out_dir)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"\nExported to {out_dir}")
    print(f"Set {kind}.bundled = true for '{args.lang}' in src/lib/languages.ts")


if __name__ == "__main__":
    main()
