

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from tqdm import tqdm
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from ultralytics import YOLO  # pip install ultralytics

# MODEL_PATH = "models/pose_landmarker_full.task"
# RAW_DIR = "raw"
# OUTPUT_CSV = "pose_dataset.csv"

# # -------------------------
# # Load YOLO detector
# # -------------------------
# def load_yolo():
#     # Load YOLOv8 pretrained model for person detection
#     model = YOLO("models/yolo11m.pt")  # tiny model; replace with yolov8s/yolov8m if needed
#     return model
# # def prepare_crop_for_mediapipe(crop):
# #     if crop is None or crop.size == 0:
# #         return None

# #     # Skip all-black crops
# #     if np.max(crop) == 0:
# #         return None

# #     # Convert BGR → RGB (if using OpenCV crop)
# #     if crop.shape[2] == 3:
# #         crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
# #     else:
# #         crop_rgb = crop

# #     # Pad to square
# #     h, w, _ = crop_rgb.shape
# #     size = max(h, w)
# #     square_crop = np.zeros((size, size, 3), dtype=np.uint8)
# #     square_crop[:h, :w] = crop_rgb

# #     return np.ascontiguousarray(square_crop, dtype=np.uint8)


# def prepare_crop_for_mediapipe(crop, fixed_height=256):
#     """
#     Prepare crop for MediaPipe:
#     - Skip all-black crops
#     - Convert BGR → RGB
#     - Resize to fixed height, preserving aspect ratio
#     - Pad width to match fixed_height if needed (optional)
#     """
#     if crop is None or crop.size == 0:
#         return None

#     if np.max(crop) == 0:  # skip all-black images
#         return None

#     # Convert BGR → RGB
#     if crop.shape[2] == 3:
#         crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#     else:
#         crop_rgb = crop

#     h, w, _ = crop_rgb.shape
#     scale = fixed_height / h
#     new_w = int(w * scale)
#     resized_crop = cv2.resize(crop_rgb, (new_w, fixed_height))

#     # Optional: pad width to make square
#     size = max(fixed_height, new_w)
#     square_crop = np.zeros((size, size, 3), dtype=np.uint8)
#     square_crop[:fixed_height, :new_w] = resized_crop

#     return np.ascontiguousarray(square_crop, dtype=np.uint8)

# # -------------------------
# # Load MediaPipe PoseLandmarker
# # -------------------------
# def load_pose_detector():
#     base = python.BaseOptions(model_asset_path=MODEL_PATH)
#     options = vision.PoseLandmarkerOptions(
#         base_options=base,
#         output_segmentation_masks=False,
#         num_poses=1
#     )
#     return vision.PoseLandmarker.create_from_options(options)

# # -------------------------
# # Extract pose keypoints
# # -------------------------
# def extract_pose(img, detector):
#     mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
#     result = detector.detect(mp_img)

#     if not result.pose_landmarks:
#         return None, None

#     keypoints = []
#     for lm in result.pose_landmarks[0]:
#         keypoints.extend([lm.x, lm.y, lm.z])
#     return keypoints, result

# # -------------------------
# # Crop person from image using YOLO bbox
# # -------------------------
# def crop_person(img, yolo_model):
#     results = yolo_model(img)
#     max_area = 0
#     best_crop = None

#     H, W, _ = img.shape

#     for r in results:
#         for box in r.boxes:
#             cls = int(box.cls[0])
#             if cls == 0:  # person class in COCO
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 area = (x2 - x1) * (y2 - y1)

#                 if area > max_area:
#                     max_area = area

#                     # --- padding 0.25 of box size ---
#                     w = x2 - x1
#                     h = y2 - y1
#                     pad_x = int(w * 0.25)
#                     pad_y = int(h * 0.25)

#                     # expanded coordinates
#                     nx1 = max(0, x1 - pad_x)
#                     ny1 = max(0, y1 - pad_y)
#                     nx2 = min(W, x2 + pad_x)
#                     ny2 = min(H, y2 + pad_y)

#                     best_crop = img[ny1:ny2, nx1:nx2]

#     return best_crop

# # def test_plot_pose(result, img):

# #     if not result.pose_landmarks:
# #         print("❌ No pose detected in test image.")
# #         return

# #     # Draw points
# #     h, w, _ = img.shape
# #     for lm in result.pose_landmarks[0]:
# #         cx, cy = int(lm.x * w), int(lm.y * h)
# #         cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)


# #     return img


# import cv2

# POSE_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,7),
#     (0,4),(4,5),(5,6),(6,8),
#     (9,10),
#     (11,12),
#     (11,13),(13,15),
#     (12,14),(14,16),
#     (15,17),(15,19),(15,21),
#     (16,18),(16,20),(16,22),
#     (11,23),(12,24),
#     (23,24),
#     (23,25),(25,27),(27,29),(29,31),
#     (24,26),(26,28),(28,30),(30,32)
# ]

# def test_plot_pose(result, img):

#     if not result.pose_landmarks:
#         print("❌ No pose detected in test image.")
#         return img

#     h, w, _ = img.shape
#     landmarks = result.pose_landmarks[0]

#     # Draw points
#     for lm in landmarks:
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(img, (cx, cy), 4, (0,255,0), -1)

#     # Draw connections
#     for s, e in POSE_CONNECTIONS:
#         x1 = int(landmarks[s].x * w)
#         y1 = int(landmarks[s].y * h)
#         x2 = int(landmarks[e].x * w)
#         y2 = int(landmarks[e].y * h)

#         cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)

#     return img
# # -------------------------
# # Build dataset
# # -------------------------
# def build_dataset():
#     yolo_model = load_yolo()
#     pose_detector = load_pose_detector()
#     rows = []

#     for class_name in os.listdir(RAW_DIR):
#         class_path = os.path.join(RAW_DIR, class_name)
#         if not os.path.isdir(class_path):
#             continue

#         for vfolder in os.listdir(class_path):
#             vpath = os.path.join(class_path, vfolder)

#             for filename in tqdm(os.listdir(vpath), desc=f"Processing {class_name}/{vfolder}"):
#                 if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
#                     continue

#                 img_path = os.path.join(vpath, filename)
#                 img = cv2.imread(img_path)
#                 # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
#                 # -------------------------
#                 # Detect person and crop
#                 # -------------------------
#                 person_crop = crop_person(img, yolo_model)

                


#                 if person_crop is None or person_crop.size == 0:
#                     continue

#                 # -------------------------
#                 # Extract pose keypoints
#                 # -------------------------
#                 person_crop = prepare_crop_for_mediapipe(person_crop, fixed_height=256)
                
#                 pose, result = extract_pose(person_crop, pose_detector)

#                 # if result is not None:
#                 #     img_plt = test_plot_pose(result, person_crop)
#                 #     cv2.imshow('Debug Crop', img_plt)
#                 #     while True:
#                 #         key = cv2.waitKey(0) & 0xFF
#                 #         if key == ord('q'):
#                 #             break
#                 #     cv2.destroyWindow('Debug Crop')
                
#                 if pose:
#                     rows.append([class_name] + pose)
#                 else:
#                     print("No Pose")

#     pose_detector.close()
#     df = pd.DataFrame(rows)
#     df.to_csv(OUTPUT_CSV, index=False)
#     print("Saved →", OUTPUT_CSV)

# if __name__ == "__main__":
#     build_dataset()







import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import random

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO  # pip install ultralytics

MODEL_PATH = "models/pose_landmarker_full.task"
RAW_DIR = "raw"
OUTPUT_CSV = "pose_dataset.csv"

# -------------------------
# Load YOLO detector
# -------------------------
def load_yolo():
    # Load YOLOv8 pretrained model for person detection
    model = YOLO("models/yolo11m.pt")  # tiny model; replace with yolov8s/yolov8m if needed
    return model

# -------------------------
# Prepare crop for MediaPipe
# -------------------------
def prepare_crop_for_mediapipe(crop, fixed_height=256):
    if crop is None or crop.size == 0:
        return None

    if np.max(crop) == 0:  # skip all-black images
        return None

    # Convert BGR → RGB
    if crop.shape[2] == 3:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = crop

    h, w, _ = crop_rgb.shape
    scale = fixed_height / h
    new_w = int(w * scale)
    resized_crop = cv2.resize(crop_rgb, (new_w, fixed_height))

    # Pad width to make square
    size = max(fixed_height, new_w)
    square_crop = np.zeros((size, size, 3), dtype=np.uint8)
    square_crop[:fixed_height, :new_w] = resized_crop

    return np.ascontiguousarray(square_crop, dtype=np.uint8)

# -------------------------
# Load MediaPipe PoseLandmarker
# -------------------------
def load_pose_detector():
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)

# -------------------------
# Extract pose keypoints
# -------------------------
def extract_pose(img, detector):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result = detector.detect(mp_img)

    if not result.pose_landmarks:
        return None, None

    keypoints = []
    for lm in result.pose_landmarks[0]:
        keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints, result

# -------------------------
# Crop person from image using YOLO bbox
# -------------------------
def crop_person(img, yolo_model):
    results = yolo_model(img)
    max_area = 0
    best_crop = None

    H, W, _ = img.shape

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area

                    # padding 0.25 of box size
                    w = x2 - x1
                    h = y2 - y1
                    pad_x = int(w * 0.25)
                    pad_y = int(h * 0.25)

                    nx1 = max(0, x1 - pad_x)
                    ny1 = max(0, y1 - pad_y)
                    nx2 = min(W, x2 + pad_x)
                    ny2 = min(H, y2 + pad_y)

                    best_crop = img[ny1:ny2, nx1:nx2]

    return best_crop

# -------------------------
# Draw pose on image (optional)
# -------------------------
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (15,17),(15,19),(15,21),
    (16,18),(16,20),(16,22),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

def test_plot_pose(result, img):
    if not result.pose_landmarks:
        return img

    h, w, _ = img.shape
    landmarks = result.pose_landmarks[0]

    # Draw points
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0,255,0), -1)

    # Draw connections
    for s, e in POSE_CONNECTIONS:
        x1 = int(landmarks[s].x * w)
        y1 = int(landmarks[s].y * h)
        x2 = int(landmarks[e].x * w)
        y2 = int(landmarks[e].y * h)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)

    return img

# -------------------------
# Build dataset with balancing
# -------------------------
def build_dataset(balance=True, max_per_class=None):
    yolo_model = load_yolo()
    pose_detector = load_pose_detector()
    rows = defaultdict(list)

    for class_name in os.listdir(RAW_DIR):
        class_path = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for vfolder in os.listdir(class_path):
            vpath = os.path.join(class_path, vfolder)

            for filename in tqdm(os.listdir(vpath), desc=f"Processing {class_name}/{vfolder}"):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(vpath, filename)
                img = cv2.imread(img_path)

                # Crop person
                person_crop = crop_person(img, yolo_model)
                if person_crop is None or person_crop.size == 0:
                    continue

                person_crop = prepare_crop_for_mediapipe(person_crop, fixed_height=256)
                pose, result = extract_pose(person_crop, pose_detector)

                if pose:
                    rows[class_name].append([class_name] + pose)
                else:
                    print(f"No Pose detected: {img_path}")

    pose_detector.close()

    # -------------------------
    # Balance dataset
    # -------------------------
    if balance:
        if max_per_class is None:
            min_count = min(len(v) for v in rows.values())
        else:
            min_count = max_per_class

        balanced_rows = []
        for cls, cls_rows in rows.items():
            if len(cls_rows) > min_count:
                balanced_rows.extend(random.sample(cls_rows, min_count))
            elif len(cls_rows) < min_count:
                # Oversample underrepresented classes
                oversample = random.choices(cls_rows, k=min_count - len(cls_rows))
                balanced_rows.extend(cls_rows + oversample)
            else:
                balanced_rows.extend(cls_rows)
        df = pd.DataFrame(balanced_rows)
    else:
        # merge all rows
        all_rows = []
        for cls_rows in rows.values():
            all_rows.extend(cls_rows)
        df = pd.DataFrame(all_rows)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV} (Balanced: {balance})")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    build_dataset(balance=True)