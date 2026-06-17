
from ultralytics import YOLO, RTDETR
from core.kalman import KalmanManager
import numpy as np
import cv2
class YOLODetector:
    def __init__(self):
        self.model = YOLO("models/yolo11m.pt")
        # YOLO class names we keep
        self.allowed = {
            "sports ball": "ball",
            "tennis racket": "racket",
            "person": "person"
        }

    def detect(self, frame, class_filter=None):
        """
        class_filter: list of strings, e.g., ["ball", "person"]
        """
        result = self.model(frame, conf=0.5, verbose=False)[0]
        all_detections = []
        person_detections = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = self.model.names[cls_id]

            # 1. Check if the object is in our allowed dictionary
            if raw_label not in self.allowed:
                continue
            
            mapped_label = self.allowed[raw_label]

            # 2. Apply the dynamic class filter
            if class_filter and mapped_label not in class_filter:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            center_x = (x1 + x2) / 2
            
            det_data = {
                "class": mapped_label,
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "center_x": center_x
            }

            if mapped_label == "person":
                person_detections.append(det_data)
            else:
                all_detections.append(det_data)

        # --- LOGIC: SELECT ONLY THE MAIN (BIGGEST) PERSON ---
        if person_detections:
            # If "person" is requested in the filter, only return the biggest one
            biggest_person = max(person_detections, key=lambda x: x['area'])
            all_detections.append(biggest_person)

        return all_detections


class RTDETRDetector:
    """
    RT-DETR on a region (e.g. person crop). COCO "sports ball" → ball, "tennis racket" → racket.
    Weights default to Ultralytics rtdetr-l.pt (downloaded on first use if missing).
    """

    def __init__(self, weights="models/rtdetr-l.pt", conf=0.6):
        self.model = RTDETR(weights)
        self.conf = conf
        self.allowed = {
            "sports ball": "ball",
            "tennis racket": "racket",
        }

    def detect(self, frame, class_filter=None):
        """
        class_filter: e.g. ["ball"], ["racket"], or ["ball", "racket"]. None = all allowed classes.
        """
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        all_detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = self.model.names[cls_id]
            if raw_label not in self.allowed:
                continue
            mapped = self.allowed[raw_label]
            if class_filter is not None and mapped not in class_filter:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            center_x = (x1 + x2) / 2

            all_detections.append(
                {
                    "class": mapped,
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                    "area": area,
                    "center_x": center_x,
                }
            )

        return all_detections


# Backward-compatible name (ball + racket both live on RTDETRDetector)
RTDETRBallDetector = RTDETRDetector


class YOLOBallDetector:
    def __init__(self):
        self.model = YOLO("models/best.pt")
        self.allowed = {
            "tennis-ball": "ball"
        }

    def detect(self, frame):
        result = self.model(frame, conf=0.2, verbose=False)[0]
    
        all_detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label not in self.allowed:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            center_x = (x1 + x2) / 2

            det = {
                "class": self.allowed[label],
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "center_x": center_x
            }

            all_detections.append(det)

        return all_detections

