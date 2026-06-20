# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import pickle

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

# MODEL_PATH = "models/pose_landmarker_full.task"

# # Load pose model
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=False,
#     num_poses=1
# )
# detector = vision.PoseLandmarker.create_from_options(options)

# # Load classifier
# model = tf.keras.models.load_model("models/tennis_pose_classifier.h5")
# label_enc = pickle.load(open("models/labels.pkl", "rb"))

# def debug_show_crop(crop):
#     if crop is None or crop.size == 0:
#         print("Empty crop!")
#         return
#     cv2.imshow("CROP DEBUG", crop)
#     cv2.waitKey(10)



# def extract_pose(img):
#     mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
#     # debug_show_crop(mp_img.numpy_view())

    
    

#     result = detector.detect(mp_img)

#     if len(result.pose_landmarks) == 0:
#         return None

#     lm = result.pose_landmarks[0]

#     keypoints = []
#     for p in lm:
#         keypoints.extend([p.x, p.y, p.z])

#     return np.array(keypoints).reshape(1, -1)


# # Confidence below this → return "unknown" or safe macro (avoid misclassification)
# # For robust shot-type decisions we prefer to be conservative; values below
# # this are treated as unreliable and mapped to a macro "unknown".
# CONFIDENCE_THRESHOLD = 0.65

# # Macro fallback when confidence is low (safe high-level category)
# MACRO_UNKNOWN = "unknown"


# def predict(img_rgb):
#     """Legacy: returns class name only. Prefer predict_proba for shot analysis."""
#     shot_type, confidence, _ = predict_proba(img_rgb)
#     return shot_type if shot_type else "Pose not detected"


# def predict_proba(img_rgb):
#     """
#     Run pose-based classifier. Returns (class_name, confidence, probs_dict).
#     - class_name: predicted class or MACRO_UNKNOWN if confidence < threshold or pose not detected.
#     - confidence: 0..1 (max probability, or 0 if no pose).
#     - probs_dict: {class_name: probability} for all classes (for averaging over frames).
#     """
#     vec = extract_pose(img_rgb)
#     if vec is None:
#         return (MACRO_UNKNOWN, 0.0, {})

#     pred = model.predict(vec, verbose=0)[0]  # 1D array of probabilities
#     class_indices = np.arange(len(pred))
#     class_names = label_enc.inverse_transform(class_indices)
#     probs_dict = {name: float(pred[i]) for i, name in enumerate(class_names)}

#     best_idx = int(np.argmax(pred))
#     confidence = float(pred[best_idx])
#     class_name = class_names[best_idx]

#     if confidence < CONFIDENCE_THRESHOLD:
#         class_name = MACRO_UNKNOWN

#     return (class_name, confidence, probs_dict)




import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Force CPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -----------------------------
# Paths
# -----------------------------
POSE_MODEL_PATH = "classification/models_c/pose_landmarker_full.task"
CLASSIFIER_PATH = "classification/models_c/tennis_pose_classifier.h5"
LABEL_PATH = "classification/models_c/labels.pkl"
SCALER_PATH = "classification/models_c/scaler.pkl"

# -----------------------------
# Load MediaPipe Pose Model
# -----------------------------
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)

# -----------------------------
# Load classifier
# -----------------------------
import tensorflow as tf

model = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)

label_enc = pickle.load(open(LABEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

class_names = label_enc.classes_

print("Loaded classes:", class_names)

# -----------------------------
# Settings
# -----------------------------
CONFIDENCE_THRESHOLD = 0.5
MACRO_UNKNOWN = "unknown"

# -----------------------------
# Debug crop viewer
# -----------------------------
def debug_show_crop(crop):
    if crop is None or crop.size == 0:
        print("Empty crop!")
        return
    cv2.imshow("CROP DEBUG", crop)
    cv2.waitKey(10)

# -----------------------------
# Pose Extraction
# -----------------------------
def extract_pose(img):

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img
    )

    result = detector.detect(mp_img)

    if len(result.pose_landmarks) == 0:
        return None

    lm = result.pose_landmarks[0]

    keypoints = []

    for p in lm:
        keypoints.extend([p.x, p.y, p.z])
    vec = np.array(keypoints).reshape(1, -1)

    return vec, result


# -----------------------------
# Prediction (simple)
# -----------------------------
def predict(img_rgb):

    shot_type, confidence, _ = predict_proba(img_rgb)

    if shot_type == MACRO_UNKNOWN:
        return "Pose not detected"

    return shot_type


# -----------------------------
# Prediction with probabilities
# -----------------------------
def predict_proba(img_rgb):
    """
    Pipeline must match training:
    - Input: RGB image already prepared like dataset (person crop + prepare_crop_for_mediapipe(crop, fixed_height=256)).
    - Pose: MediaPipe full 33 landmarks → 99 values (x,y,z per landmark).
    - Scaler: same StandardScaler as in train.py.
    """
    out = extract_pose(img_rgb)
    if out is None:
        return (MACRO_UNKNOWN, 0.0, {})
    vec, _ = out

    # Ensure feature dim matches training (99 = 33 landmarks * 3)
    n_features = scaler.n_features_in_
    if vec.shape[1] != n_features:
        raise ValueError(
            f"Infer pose features {vec.shape[1]} != scaler/train features {n_features}. "
            "Pipeline must match dataset.py (same pose extraction)."
        )

    # Apply scaler (VERY IMPORTANT)
    vec = scaler.transform(vec)

    # Model prediction
    pred = model.predict(vec, verbose=0)[0]

    probs_dict = {
        class_names[i]: float(pred[i])
        for i in range(len(pred))
    }

    best_idx = int(np.argmax(pred))
    confidence = float(pred[best_idx])
    class_name = class_names[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        class_name = MACRO_UNKNOWN

    return (class_name, confidence, probs_dict)