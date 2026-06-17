# Tannis App

End-to-end **video analysis pipeline** for padel/tennis: detect player, racket, and ball; find racket–ball impacts; classify each shot from body pose; compute per-shot metrics and rule-based coaching scores. Results are written to an annotated video and a structured JSON file.

---

## What it does

| Stage | Technology | Output |
|-------|------------|--------|
| Object detection | YOLO11 (person) + RT-DETR (ball/racket on player crop) | Bounding boxes per frame |
| Tracking | ByteTrack + Kalman filtering | Stable ball/racket trajectories |
| Impact detection | Proximity of racket and ball (+ wrist pose when available) | Impact frame index |
| Shot classification | MediaPipe pose + trained Keras classifier | Shot type + confidence |
| Analysis | Rule-based metrics and scoring | Score, reasons, tips per shot |

---

## Project structure

```
tannisapp/
├── app.py                    # Entry point: process a video file
├── core/
│   ├── pipeline.py           # Main vision pipeline (detection → impact → classify)
│   ├── detector.py           # YOLO / RT-DETR wrappers
│   ├── infer.py              # Pose extraction + shot classifier
│   ├── shot_analysis.py      # Metrics, scoring, JSON schema
│   ├── tracker.py            # ByteTrack
│   ├── kalman.py             # Ball/racket smoothing
│   ├── court_lines.py        # Court line overlay
│   └── overlay.py            # HUD / debug drawing
├── classification/           # Dataset build + model training (see classification/README.md)
├── models/                   # YOLO / RT-DETR weights (not in git)
├── storage/
│   └── json_writer.py        # Save results.json
├── docs/
│   └── shot_schema.md        # Full JSON schema reference
└── output/                   # Generated videos and JSON (not in git)
```

---

## Requirements

- Python 3.10+ recommended
- Webcam not required; processes video files with OpenCV

### Install

```bash
git clone https://github.com/MNaseerSubhani/tennisapp
cd tannisapp

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install tensorflow pandas scikit-learn tqdm
```

`tensorflow` is required for shot classification at inference time. The other packages are needed if you [train the classifier](classification/README.md).

---

## Model weights (before first run)

Large files are **not** committed to git. Set up weights as follows.

### 1. Detection models (`models/` at repo root)

| File | Used by | Notes |
|------|---------|--------|
| `yolo11m.pt` | `core/detector.py` | Person detection; auto-downloaded by Ultralytics on first run |
| `rtdetr-l.pt` | `core/detector.py` | Ball and racket on player crop; auto-downloaded on first run |
| `best.pt` | Optional custom ball detector | Only if you switch to `YOLOBallDetector` |

Create the folder and run the pipeline once; missing Ultralytics weights are fetched automatically:

```bash
mkdir models
python app.py
```

### 2. Classification models (`classification/models/`)

| File | Notes |
|------|--------|
| `pose_landmarker_full.task` | [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — **full** model |
| `yolo11m.pt` | Same as above; used when building the training dataset |
| `tennis_pose_classifier.h5` | Trained classifier — run [training](classification/README.md) or use a provided checkpoint |
| `labels.pkl`, `scaler.pkl` | Saved with `train.py` |

---

## Inference (step by step)

### Step 1 — Prepare a video

Place your input file anywhere accessible, e.g. `test/my_match.mp4`. The `test/` folder is gitignored so you can keep local videos there.

### Step 2 — Run the pipeline

**Default** (edit the path in `app.py` or call from code):

```bash
python app.py
```

`app.py` currently points at `test/vid2.mp4`. Change `input_video_path` in `app.py`, or import the function:

```python
from app import process_video

process_video("path/to/your_video.mp4", output_path="output")
```

### Step 3 — What happens internally

1. **Load models** — YOLO, RT-DETR, MediaPipe pose, Keras classifier (`core/infer.py`).
2. **Per frame** — Detect person; track ball and racket; draw court lines and HUD.
3. **On impact** — When racket and ball are close enough, record impact frame and buffer ±2 frames for classification.
4. **Classify** — Crop player, extract pose, scale features, predict shot type; aggregate over the window.
5. **Score** — Compute metrics (contact height, distance to body, zone, etc.) and rule-based score with tips.
6. **Write outputs** — Annotated side-by-side video and `results.json`.

### Step 4 — Read the outputs

| File | Description |
|------|-------------|
| `output/output.mp4` | Original frame + HUD panel (detections, impact marker, shot info) |
| `output/results.json` | **Main result**: one object per detected impact |

Console summary example:

```
Processing complete. Output video saved to output, 3 impacts to output/results.json
```

### Step 5 — Integrate JSON

Each element in `results.json` contains:

- `shot_type` — e.g. `Volea`, or `unknown` if confidence is low
- `confidence` — 0–1
- `quality_flag` — `high` | `medium` | `low`
- `metrics` — impact frame, window, heights, distances, zone, etc.
- `scoring` — score 0–100, `score_reasons`, `tips`

Full field reference: [docs/shot_schema.md](docs/shot_schema.md).

**Example (abbreviated):**

```json
[
  {
    "shot_type": "Volea",
    "confidence": 0.72,
    "quality_flag": "high",
    "metrics": {
      "impact_time": 42,
      "shot_window": { "start": 40, "end": 44 },
      "impact_height": 0.48,
      "player_zone": "center"
    },
    "scoring": {
      "score": 85,
      "score_reasons": ["impact_height_in_ideal_range"],
      "tips": ["Recover to ready position after each shot."]
    }
  }
]
```

---

## Training the shot classifier (step by step)

Training is documented in detail in **[classification/README.md](classification/README.md)**. Short version:

```bash
# 1. Add labeled stills: classification/raw/<ShotType>/<Session>/*.PNG

# 2. Build pose feature CSV
cd classification
python dataset.py

# 3. Train and save weights
python train.py

# 4. Inference uses new weights automatically
cd ..
python app.py
```

After training, these files must exist:

- `classification/models/tennis_pose_classifier.h5`
- `classification/models/labels.pkl`
- `classification/models/scaler.pkl`

---

## Configuration tips

| Setting | Location | Default |
|---------|----------|---------|
| Impact classification window | `core/pipeline.py` — `SHOT_WINDOW_HALF` | ±2 frames around impact |
| Classification confidence cutoff | `core/infer.py` — `CONFIDENCE_THRESHOLD` | `0.5` |
| YOLO person confidence | `core/detector.py` | `0.5` |
| RT-DETR ball/racket confidence | `core/detector.py` | `0.6` |

Lower the confidence threshold if valid shots are labeled `unknown` too often; raise it to reduce false classifications.

---

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| `Could not open video` | Path in `app.py`; codec support (try H.264 MP4) |
| No impacts in JSON | Ball/racket not detected; try a clearer angle or lighting |
| All shots `unknown` | Train or copy classifier weights; ensure player is visible in crop |
| `FileNotFoundError` for models | See [Model weights](#model-weights-before-first-run) |
| Slow processing | First run downloads weights; CPU inference is expected without GPU |

---

## License and data

Training images under `classification/raw/` are local assets and excluded from git. Do not commit large videos, weights, or generated CSVs — see `.gitignore`.
