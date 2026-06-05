# Attribution & third-party assets

The project source code is MIT licensed (see LICENSE). The bundled assets and
pretrained model weights below come from third parties and carry their own
terms.

## Runtime libraries / models

- **MediaPipe Tasks — HandLandmarker** (`public/models/hand_landmarker.task`,
  `public/models/wasm/*`) — © Google, Apache License 2.0.
  https://developers.google.com/mediapipe
- **TensorFlow.js** — © Google, Apache License 2.0.
- **@mediapipe/tasks-vision** — Apache License 2.0.

## Pretrained sign-language models (derivatives of public datasets)

The alphabet models in `public/models/static/<lang>/` were trained by this
project on the datasets below. They contain only learned weights over
MediaPipe hand-landmark coordinates (not the original images). Each dataset's
license governs its use — **verify on the dataset page before redistribution**.

| Model | Dataset | Source |
|---|---|---|
| `static/asl` (ASL A–Z) | ASL Alphabet | Kaggle `grassknoted/asl-alphabet` |
| `static/sibi` (SIBI A–Y) | SIBI Dataset | Kaggle `alvinbintang/sibi-dataset` |
| `static/bisindo` (BISINDO A–Z) | Indonesian Sign Language (BISINDO) | Kaggle `agungmrf/indonesian-sign-language-bisindo` |

If any dataset's license does not permit redistributing derived weights, remove
the corresponding folder under `public/models/static/` and set its
`static.bundled = false` in `src/lib/languages.ts`. Users can then regenerate
the model locally with the scripts in `training/` (see `training/README.md`).

## Notes

- Trained models are tiny MLP/GRU classifiers over 21-keypoint hand landmarks;
  the original dataset imagery is not included in this repository.
- To reproduce or retrain any model, see `training/`.
