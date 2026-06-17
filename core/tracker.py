

import supervision as sv
import numpy as np

# class ByteTracker:
#     def __init__(self):
#         self.tracker = sv.ByteTrack()

#     def update(self, detections):
#         if not detections:
#             return []
#         xyxy = np.array([d["bbox"] for d in detections])
#         conf = np.array([d["confidence"] for d in detections])
#         cls = np.zeros(len(detections))
#         det = sv.Detections(
#             xyxy=xyxy,
#             confidence=conf,
#             class_id=cls
#         )
#         tracked = self.tracker.update_with_detections(det)
#         return [
#             {
#                 "track_id": int(tracked.tracker_id[i]),
#                 "bbox": tracked.xyxy[i].tolist()
#             }
#             for i in range(len(tracked))
#         ]

class BallTracker:
    def __init__(self, max_missed=8, dist_th=60):
        self.tracks = {}
        self.next_id = 1
        self.max_missed = max_missed
        self.dist_th = dist_th

    def _center(self, bbox):
        return np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])

    def update(self, detections, frame_id):
        results = []
        used_dets = set()

        # Predict all tracks
        predictions = {
            tid: self.tracks[tid].predict()
            for tid in self.tracks
        }

        # Associate detections → tracks (distance-based)
        for det_i, det in enumerate(detections):
            det_center = self._center(det["bbox"])

            best_tid = None
            best_dist = float("inf")

            for tid, pred in predictions.items():
                pred_center = pred[:2]
                dist = np.linalg.norm(det_center - pred_center)

                if dist < best_dist and dist < self.dist_th:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid].update(det["bbox"], frame_id)
                used_dets.add(det_i)

        # Mark missed
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].last_seen != frame_id:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        # Create new tracks
        for i, det in enumerate(detections):
            if i in used_dets:
                continue
            self.tracks[self.next_id] = BallTrack(
                self.next_id,
                det["bbox"],
                kalman=KalmanManager(),
                frame_id=frame_id
            )
            self.next_id += 1

        # Output
        for tid, track in self.tracks.items():
            state = track.kalman.get_state()
            if state is None:
                continue
            x, y = state[:2]
            results.append({
                "track_id": tid,
                "bbox": track.kalman.get_bbox(),
                "class": "ball"
            })

        return results


class ByteTracker:
    def __init__(self):
        self.trackers = {
            "person": sv.ByteTrack(),
            "racket": sv.ByteTrack(),
            "ball": sv.ByteTrack()
        }

    def update(self, detections):
        results = []

        for cls_name, tracker in self.trackers.items():
            cls_dets = [d for d in detections if d["class"] == cls_name]
            if not cls_dets:
                continue

            det = sv.Detections(
                xyxy=np.array([d["bbox"] for d in cls_dets]),
                confidence=np.array([d["confidence"] for d in cls_dets]),
                class_id=np.zeros(len(cls_dets))
            )

            tracks = tracker.update_with_detections(det)

            for i in range(len(tracks)):
                results.append({
                    "track_id": int(tracks.tracker_id[i]),
                    "bbox": tracks.xyxy[i].tolist(),
                    "class": cls_name
                })

        return results
