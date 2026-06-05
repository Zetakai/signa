# Training pipeline — produce bundled models

Turn a dataset into a TF.js model the web app ships by default, so users just
**pick a language** instead of training their own.

```bash
cd training
python -m venv .venv && source .venv/bin/activate   # Python 3.10–3.11
pip install -r requirements.txt
```

> Note: `mediapipe` + `tensorflowjs` have the best wheel support on Python
> 3.10/3.11. On 3.12+ you may hit install issues.

## Option 1 — from an image dataset (letters / static)

Lay out images one folder per label:

```
dataset/
  A/ *.jpg
  B/ *.jpg
  ...
```

Good public sources (download manually, check license):
- ASL alphabet: Kaggle `grassknoted/asl-alphabet`
- SIBI (Indonesia) alphabet: search Kaggle "SIBI alphabet"

```bash
python extract_and_train_static.py --dataset ./dataset --lang asl --epochs 40
```

Outputs `public/models/static/asl/{model.json, *.bin, labels.json}`.

## Option 2 — from in-app exported data (letters or words)

Collect samples on real devices in the app's **Train** tab, press **Export**,
then bake them into a shared model:

```bash
python train_from_export.py --input signa-static-samples.json  --lang asl
python train_from_export.py --input signa-dynamic-samples.json --lang asl --epochs 60
```

Outputs `public/models/<static|dynamic>/<lang>/…`.

## After training — make the model load by default

Edit `src/lib/languages.ts` and flip the slot to bundled, e.g.:

```ts
{ id: 'asl', name: 'American Sign Language',
  static: { bundled: true }, dynamic: { bundled: false } }
```

Then `npm run build` and commit the new files under `public/models/`. On next
load the app finds the model and the picker shows **"Model ready"** — no user
training needed.

## Must stay in sync with the app

The scripts mirror the app exactly — keep them aligned if you change either:
- normalization: `src/lib/normalize.ts` ↔ `normalize_hand()`
- static MLP arch: `src/lib/staticModel.ts`
- dynamic GRU arch + `DYNAMIC_TIMESTEPS=24`: `src/lib/dynamicModel.ts`, `src/lib/sequenceBuffer.ts`
