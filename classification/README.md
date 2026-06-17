# Shot classification (pose-based)

This module trains a **padel/tennis shot-type classifier** from still images. Each image is cropped to the player (YOLO), pose keypoints are extracted (MediaPipe), and a small neural network learns to map the 99-dimensional pose vector to a shot label.

The trained model is used at inference time by `core/infer.py` inside the main video pipeline.

## Shot classes

Organize training images under `raw/<class_name>/<video_or_session>/`. Current classes in the dataset:

| Folder | Shot type |
|--------|-----------|
| `Bandeja` | Bandeja |
| `Globo` | Lob (globo) |
| `GolpeDeFondo` | Groundstroke |
| `Reves` | Backhand |
| `Saque` | Serve |
| `Smash` | Smash |
| `Vibora` | Víbora |
| `Volea` | Volley |

Add or rename folders to match the labels you want the model to predict.

## Directory layout

```
classification/
├── raw/                          # Training images (not in git — see root .gitignore)
│   └── Volea/
│       └── V1/
│           ├── 1.PNG
│           └── 2.PNG
├── models/                       # Weights and trained artifacts
│   ├── pose_landmarker_full.task # MediaPipe pose (required)
│   ├── yolo11m.pt                # Person detector for cropping (required)
│   ├── tennis_pose_classifier.h5 # Trained classifier (output of train.py)
│   ├── labels.pkl                # Label encoder (output of train.py)
│   └── scaler.pkl                # Feature scaler (output of train.py)
├── dataset.py                    # Step 1: images → pose_dataset.csv
├── train.py                      # Step 2: CSV → trained model
└── pose_dataset.csv              # Generated feature table (not in git)
```

**Image naming:** any `.png`, `.jpg`, or `.jpeg` file inside a session folder is used. The parent folder name (`Volea`, `Saque`, …) is the class label.

---

## Prerequisites

From the **repository root**, create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install tensorflow pandas scikit-learn tqdm
```

### Required model files

Place these under `classification/models/` before building the dataset:

| File | Purpose | How to obtain |
|------|---------|---------------|
| `pose_landmarker_full.task` | MediaPipe full-body pose | [MediaPipe pose landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — download the **full** float model |
| `yolo11m.pt` | Person detection for crops | Downloaded automatically by Ultralytics on first run, or copy from your Ultralytics cache |

---

## Step 1 — Build the pose dataset

All commands below are run from the **`classification/`** directory:

```bash
cd classification
python dataset.py
```

**What it does:**

1. Walks every image under `raw/<class>/<session>/`.
2. Detects the largest person with YOLO (`yolo11m.pt`).
3. Crops with 25% padding, resizes to height 256 px, pads to square.
4. Runs MediaPipe Pose Landmarker → **99 features** (33 landmarks × x, y, z).
5. Balances classes (undersample majority, oversample minority).
6. Writes `pose_dataset.csv` (column 0 = label, columns 1–99 = keypoints).

**Options** (edit `dataset.py` or call from code):

```python
from dataset import build_dataset

build_dataset(balance=True)           # default: balance to smallest class count
build_dataset(balance=True, max_per_class=200)  # cap samples per class
build_dataset(balance=False)          # keep all extracted poses
```

**Expected output:**

```
Processing Volea/V1: 100%|██████████| 12/12
...
Saved → pose_dataset.csv (Balanced: True)
```

If many images print `No Pose detected`, check lighting, full-body visibility, and that `pose_landmarker_full.task` is present.

---

## Step 2 — Train the classifier

```bash
python train.py
```

**What it does:**

1. Loads `pose_dataset.csv`.
2. Encodes labels and applies `StandardScaler` to features.
3. Augments data (Gaussian noise + horizontal flip of x-coordinates).
4. Trains a small dense Keras model (128 → 64 → softmax).
5. Evaluates on a held-out 10% test split.
6. Saves artifacts to `models/`:
   - `tennis_pose_classifier.h5`
   - `labels.pkl`
   - `scaler.pkl`

**Expected output:**

```
Train: (N, 99)
Test: (M, 99)
...
Test Accuracy: 0.85
Saved:
 - tennis_pose_classifier.h5
 - labels.pkl
 - scaler.pkl
```

Training uses CPU by default in inference (`core/infer.py` sets `CUDA_VISIBLE_DEVICES=-1`). For faster training, remove or adjust that line in `infer.py` only; `train.py` will use GPU if TensorFlow detects one.

---

## Step 3 — Verify (optional)

Quick pose visualization on a single image:

```bash
python test.py --test "raw/Volea/V1/1.PNG" --save pose_debug.jpg
```

This draws landmarks on the image and saves `pose_debug.jpg`. It does **not** run the trained classifier; use the main pipeline for end-to-end inference.

---

## Retraining workflow (summary)

```bash
# 1. Add images to raw/<Class>/<Session>/
# 2. Rebuild features
cd classification
python dataset.py

# 3. Retrain
python train.py

# 4. Run video inference from repo root (picks up new weights automatically)
cd ..
python app.py
```

No code changes are needed after retraining as long as artifact paths stay the same and the pose pipeline matches training (same crop height, same MediaPipe model).

---

## How inference uses this model

`core/infer.py` loads:

- `classification/models/pose_landmarker_full.task`
- `classification/models/tennis_pose_classifier.h5`
- `classification/models/labels.pkl`
- `classification/models/scaler.pkl`

For each frame in the impact window, the video pipeline:

1. Crops the tracked player (same padding logic as `dataset.py`).
2. Prepares the crop (`prepare_crop_for_mediapipe`, height 256).
3. Extracts pose → scales features → predicts class probabilities.
4. Aggregates probabilities across frames and applies a confidence threshold (`0.5`); below threshold → `"unknown"`.

**Important:** Training and inference must use the **same** preprocessing. If you change crop size or pose model in `dataset.py`, update `core/pipeline.py` and `core/infer.py` to match.

---

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|----------------|-----|
| `No Pose detected` on many images | Person too small, occluded, or wrong crop | Use clearer frames; check YOLO detects a person |
| Low test accuracy | Too few images per class or similar poses | Add more diverse sessions per class |
| `Infer pose features != scaler/train features` | Pose model or landmark count changed | Rebuild dataset and retrain; keep MediaPipe **full** model |
| Wrong class at runtime | Stale weights or domain shift | Retrain with frames similar to your videos |
| `FileNotFoundError` for `.task` / `.pt` | Missing weights | Download pose model; run once to fetch YOLO weights |

---

## Adding a new shot type

1. Create `raw/<NewClass>/<Session>/` and add images.
2. Run `python dataset.py` then `python train.py`.
3. Confirm `labels.pkl` includes the new class (`Loaded classes:` printed when the pipeline starts).
4. Optionally tune `CONFIDENCE_THRESHOLD` in `core/infer.py`.
