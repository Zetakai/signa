# Signa — Sign Language Detection (web)

Detects sign language from your camera, fully **in the browser**. Static
fingerspelling (letters) and dynamic word gestures, disambiguated by a motion
gate, accumulated into a live transcript, with an optional LLM "smooth" button
that turns ASL gloss into fluent English.

Built for **Cloudflare Pages** (static site + one Pages Function).

## Stack

- **React + Vite + TypeScript** — static SPA, output `dist/`
- **MediaPipe Tasks HandLandmarker** (`@mediapipe/tasks-vision`) — 21 hand keypoints, in-browser WASM
- **TensorFlow.js** — landmark classifiers (MLP for letters, GRU for words)
- **Cloudflare Pages Function + Workers AI** — `/smooth` gloss → English

## How it works

```
webcam → HandLandmarker → 21 keypoints → normalize (63-vec)
       → motion gate:  still → letter (MLP)
                       moving → word segment (GRU)
                       hands down/pause → space
       → sentence builder → live transcript → [Smooth] → Workers AI → English
```

Inference is 100% client-side and works offline. The only backend is the
optional `/smooth` function.

## Develop

```bash
npm install
npm run dev            # http://localhost:5173
```

Test the smooth function locally (needs Workers AI binding):

```bash
npm run build
npx wrangler pages dev dist
```

## Ready languages (pick and use)

These ship with pretrained alphabet models — pick from the dropdown, no
training needed:

| Language | Letters | Hands | Held-out val acc |
|---|---|---|---|
| American Sign Language (ASL) | A–Z | one | ~99% |
| SIBI (Indonesia) | A–Y (no J/Z) | one | ~96% |
| BISINDO (Indonesia) | A–Z | two | ~96% |

Word/sentence recognition is not bundled (needs video datasets) — train words
in-app or via the pipeline. Accuracy above is on each dataset's distribution;
live webcam varies with lighting/angle. If a letter misfires, capture a few
samples in Train to fine-tune for your hand.

## Train your own signs (custom / words / new languages)

Open the **Train** tab:

1. **Letters (static):** type a letter, hold the pose, press **Capture** (records a burst). Do ~5 bursts per letter from slightly different angles.
2. **Words (dynamic):** type a word, press **Record**, perform the gesture, press **Stop**. ~5 recordings per word.
3. Press **Train model**. The model trains in-browser and saves to IndexedDB (stays on your device).
4. Switch to **Detect** — your signs are now recognized.

Optional: drop a pretrained TF.js model into `public/models/static/` or
`public/models/dynamic/` (`model.json` + `*.bin`) to ship default signs.

## Deploy to Cloudflare Pages (Git integration)

1. Push this repo to GitHub.
2. Cloudflare dashboard → **Pages → Create → Connect to Git** → pick the repo.
3. Build settings:
   - Framework preset: **Vite**
   - Build command: `npm run build`
   - Build output directory: `dist`
   - Node version: 20 (via `.nvmrc`)
4. For the **Smooth** button: Pages → **Settings → Functions → Bindings** →
   add **Workers AI**, variable name **`AI`**.
5. Deploy. Camera needs HTTPS — Pages provides it.

The `functions/smooth.ts` file auto-deploys as the `/smooth` Pages Function.

## Notes / limits

- Single-hand MVP. Two-handed signs need `numHands: 2` and a doubled feature
  vector (see `initHandLandmarker` / `normalizeHand`).
- Tune motion thresholds in `src/lib/motionGate.ts` (`DEFAULT_GATE_CONFIG`) if
  letters/words misfire for your signing speed.
- To swap the smooth model from Workers AI to the Claude API, edit
  `functions/smooth.ts`.
