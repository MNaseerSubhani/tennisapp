import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
from tqdm import tqdm
import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models_c/pose_landmarker_full.task"
RAW_DIR = "raw"
OUTPUT_CSV = "pose_dataset.csv"


# ---------------------------------------------------------
#  Load Pose Detector
# ---------------------------------------------------------
def load_detector():
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)


# ---------------------------------------------------------
#  Extract Pose (for dataset building)
# ---------------------------------------------------------
def extract_pose(img, detector):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result = detector.detect(mp_img)

    if not result.pose_landmarks:
        return None

    keypoints = []
    for lm in result.pose_landmarks[0]:
        keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints


# ---------------------------------------------------------
#   Draw Pose for Testing
# ---------------------------------------------------------
def test_plot_pose(image_path, save_path="pose_test_output.jpg"):
    detector = load_detector()

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image:", image_path)
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_img)

    if not result.pose_landmarks:
        print("❌ No pose detected in test image.")
        return

    # Draw points
    h, w, _ = img.shape
    for lm in result.pose_landmarks[0]:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

    detector.close()

    # Save instead of show
    cv2.imwrite(save_path, img)
    print(f"✅ Pose image saved → {save_path}")


# ---------------------------------------------------------
#   Dataset Builder
# ---------------------------------------------------------
def build_dataset():
    detector = load_detector()
    rows = []

    for class_name in os.listdir(RAW_DIR):
        class_path = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for vfolder in os.listdir(class_path):
            vpath = os.path.join(class_path, vfolder)

            for filename in tqdm(os.listdir(vpath),
                                 desc=f"Processing {class_name}/{vfolder}"):

                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(vpath, filename)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pose = extract_pose(img_rgb, detector)
                if pose:
                    rows.append([class_name] + pose)

    detector.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved →", OUTPUT_CSV)


# ---------------------------------------------------------
#   MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Path to test image")
    parser.add_argument("--save", type=str, default="pose_test_output.jpg",
                        help="Output file path for test mode")
    args = parser.parse_args()

    if args.test:
        test_plot_pose(args.test, args.save)
    else:
        build_dataset()
