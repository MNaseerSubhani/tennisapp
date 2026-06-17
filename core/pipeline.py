

from collections import deque
from core.detector import YOLODetector, RTDETRDetector
from core.tracker import ByteTracker
from core.kalman import KalmanManager
from core.court_lines import detect_court_lines
from core.overlay import draw
from core import shot_analysis
import numpy as np
import cv2
import core.infer as infer

# Frames before/after impact to run prediction (impact_start = impact - SHOT_WINDOW_HALF, etc.)
SHOT_WINDOW_HALF = 2
# Frames to keep overlay (impact + shot metrics) visible after impact (~1.5 s at 30 fps)
OVERLAY_LINGER_FRAMES = 45

# MediaPipe pose: right wrist; racket must be near this when pose is available
MP_RIGHT_WRIST = 16
RACKET_WRIST_VISIBILITY_MIN = 0.25
# Max distance (pixels) from racket center to wrist as a fraction of person bbox max side
RACKET_TO_WRIST_MAX_FRAC = 0.30

def is_valid_crop(crop):
    if crop is None or crop.size == 0:
        return False
    if np.max(crop) == 0:   # all pixels are zero → black
        return False
    return True


# def prepare_crop_for_mediapipe(crop):
#     if crop is None or crop.size == 0:
#         return None

#     # Skip all-black crops
#     if np.max(crop) == 0:
#         return None

#     # Convert BGR → RGB (if using OpenCV crop)
#     if crop.shape[2] == 3:
#         crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#     else:
#         crop_rgb = crop

#     # Pad to square
#     h, w, _ = crop_rgb.shape
#     size = max(h, w)
#     square_crop = np.zeros((size, size, 3), dtype=np.uint8)
#     square_crop[:h, :w] = crop_rgb

#     return np.ascontiguousarray(square_crop, dtype=np.uint8)


def prepare_crop_for_mediapipe(crop, fixed_height=256):
    """
    Prepare crop for MediaPipe:
    - Skip all-black crops
    - Convert BGR → RGB
    - Resize to fixed height, preserving aspect ratio
    - Pad width to match fixed_height if needed (optional)
    """
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

    # Optional: pad width to make square
    size = max(fixed_height, new_w)
    square_crop = np.zeros((size, size, 3), dtype=np.uint8)
    square_crop[:fixed_height, :new_w] = resized_crop

    return np.ascontiguousarray(square_crop, dtype=np.uint8)

def line_intersection(p1, v1, p2, v2):
    """Return intersection point of two parametric lines:
       L1 = p1 + t*v1
       L2 = p2 + s*v2
    """
    A = np.array([v1, -v2]).T
    b = p2 - p1

    if np.linalg.det(A) == 0:
        return None  # parallel or no intersection

    t_s = np.linalg.solve(A, b)
    t = t_s[0]

    if t < 0:
        return None  # intersection behind the ball → not meaningful

    return p1 + t * v1


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
        print("❌ No pose detected in test image.")
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

def debug_show_crop(crop):
    if crop is None or crop.size == 0:
        print("Empty crop!")
        return
    cv2.imshow("CROP DEBUG", crop)
    cv2.waitKey(1)


def right_wrist_xy_full_frame(crop_bgr, offset_x, offset_y, fixed_height=256):
    """
    Map MediaPipe right wrist to full-frame pixel coordinates.
    crop_bgr is the person crop; offset_x/offset_y is the crop's top-left in the full frame.
    Returns (x, y) or None if pose / wrist is missing or low visibility.
    """
    prepared = prepare_crop_for_mediapipe(crop_bgr, fixed_height=fixed_height)
    if prepared is None:
        return None
    out = infer.extract_pose(prepared)
    if out is None:
        return None
    _vec, result = out
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    if len(lm) <= MP_RIGHT_WRIST:
        return None
    rw = lm[MP_RIGHT_WRIST]
    vis = getattr(rw, "visibility", None)
    if vis is not None and float(vis) < RACKET_WRIST_VISIBILITY_MIN:
        return None

    h_orig, w_orig, _ = crop_bgr.shape
    size = prepared.shape[0]
    px = float(rw.x) * size
    py = float(rw.y) * size

    scale = fixed_height / float(h_orig)
    new_w = int(round(w_orig * scale))
    new_w = max(1, new_w)

    px = float(np.clip(px, 0.0, new_w - 1e-6))
    py = float(np.clip(py, 0.0, fixed_height - 1e-6))

    orig_x = (px / new_w) * w_orig
    orig_y = (py / fixed_height) * h_orig
    return (orig_x + offset_x, orig_y + offset_y)


class VisionPipeline:
    def __init__(self):
        self.detector = YOLODetector()
        self.rtdetr = RTDETRDetector()
        self.tracker = ByteTracker()
        self.kalman = KalmanManager()

        self.trajectory_histories = {} 
        self.selected_ids = {"person": None, "racket": None, "ball": None}
        self.class_tracks = {"person": [], "racket": [], "ball": []}

        # Real impact
        self.racket_ball_impact_point = None
        self.racket_ball_impact_frame = None

        # Prediction (continuous tracking, always ON)
        self.ball_racket_history = []
        self.predicted_impact_point = None  

        # Racket motion
        self.racket_previous_position = None
        self.racket_current_velocity = None

        # Shot analysis: buffer last N frames, pending impact until window is full
        self._frame_buffer = {}  # frame_id -> {crop_rgb, person_bbox, ball_bbox, racket_bbox}
        self._pending_impact = None  # (impact_frame, impact_point, person_bbox, ball_bbox, racket_bbox, frame_h, frame_w, court_lines)
        self.shots = []  # list of shot output dicts (stable JSON schema)
        self._last_shot = None  # most recently emitted shot (for this process() return)
        self._display_until_frame = None  # keep overlay visible until this frame
        self._display_impact_point = None
        self._display_impact_frame = None


    # ----------------------------------------------------------------------
    def _select_main_entities(self, tracks, detections):
        current_person_ids = []
        current_racket_ids = []
        current_ball_ids = []

        for d in detections:
            for t in tracks:
                if np.allclose(d["bbox"], t["bbox"], atol=1e-2):
                    d["track_id"] = t["track_id"]
                    if d["class"] == "person":
                        current_person_ids.append(t["track_id"])
                    elif d["class"] == "racket":
                        current_racket_ids.append(t["track_id"])
                    elif d["class"] == "ball":
                        current_ball_ids.append(t["track_id"])

        self.class_tracks = {
            "person": current_person_ids,
            "racket": current_racket_ids,
            "ball": current_ball_ids
        }

        # PERSON
        if self.selected_ids["person"] not in current_person_ids:
            pts = [t for t in tracks if t["track_id"] in current_person_ids]
            if pts:
                biggest = max(pts, key=lambda t: (t["bbox"][2]-t["bbox"][0])*(t["bbox"][3]-t["bbox"][1]))
                self.selected_ids["person"] = biggest["track_id"]
            else:
                self.selected_ids["person"] = None

        # RACKET
        if self.selected_ids["racket"] not in current_racket_ids:
            if self.selected_ids["person"]:
                pb = next((t["bbox"] for t in tracks if t["track_id"] == self.selected_ids["person"]), None)

                if pb:
                    px = (pb[0] + pb[2]) / 2
                    py = (pb[1] + pb[3]) / 2

                    def dist(rb):
                        rcx = (rb[0]+rb[2])/2
                        rcy = (rb[1]+rb[3])/2
                        return np.sqrt((rcx - px)**2 + (rcy - py)**2)

                    rackets = [t for t in tracks if t["track_id"] in current_racket_ids]
                    if rackets:
                        closest = min(rackets, key=lambda t: dist(t["bbox"]))
                        self.selected_ids["racket"] = closest["track_id"]
                    else:
                        self.selected_ids["racket"] = None
                else:
                    self.selected_ids["racket"] = None
            else:
                self.selected_ids["racket"] = None

        # BALL
        if self.selected_ids["ball"] not in current_ball_ids:
            balls = [t for t in tracks if t["track_id"] in current_ball_ids]
            filtered = []
            for b in balls:
                bb = b["bbox"]
                area = (bb[2]-bb[0])*(bb[3]-bb[1])
                v = self.kalman.get_velocity_magnitude(b["track_id"])
                if area > 0 and v is not None and v > 0:
                    filtered.append((b, area, v))

            if filtered:
                filtered.sort(key=lambda x: x[2], reverse=True)
                self.selected_ids["ball"] = filtered[0][0]["track_id"]
            else:
                self.selected_ids["ball"] = None


    # ----------------------------------------------------------------------
    def _detect_impact_from_detections(self, detections, frame_id, crop, frame_height, frame_width):
        ball_dets = [d for d in detections if d["class"] == "ball"]
        racket_dets = [d for d in detections if d["class"] == "racket"]

        if not ball_dets or not racket_dets:
            return None

        def center(b):
            return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

        min_dist = 99999
        best_point = None
        best_ball = None
        best_racket = None

        for b in ball_dets:
            bc = center(b["bbox"])
            for r in racket_dets:
                rc = center(r["bbox"])
                d = np.linalg.norm(bc - rc)
                if d < min_dist:
                    min_dist = d
                    best_point = (bc + rc) / 2
                    best_ball = b
                    best_racket = r

        if best_point is not None and best_ball is not None and best_racket is not None:
            # Compute a dynamic distance threshold based on object size:
            # - when ball/racket appear large (close), the allowed distance should be smaller
            # - when they appear small (far), the allowed distance can be larger
            def bbox_diag(bbox):
                x1, y1, x2, y2 = bbox
                return np.linalg.norm([x2 - x1, y2 - y1])

            ball_size = bbox_diag(best_ball["bbox"])
            racket_size = bbox_diag(best_racket["bbox"])
            avg_size = (ball_size + racket_size) / 2.0 if (ball_size and racket_size) else 0.0

            # base_thresh = 100.0
            # ref_size = 100.0  # heuristic reference size in pixels
            # if avg_size > 0:
            #     dynamic_thresh = base_thresh * (ref_size / avg_size)
            # else:
            #     dynamic_thresh = base_thresh

            # # Clamp to reasonable bounds
            # dynamic_thresh = float(np.clip(dynamic_thresh, 5.0, 60.0))

            if min_dist < 80 and self._pending_impact is None:
                self.racket_ball_impact_point = tuple(best_point.astype(int))
                self.racket_ball_impact_frame = frame_id
                person_det = next((d for d in detections if d["class"] == "person"), None)
                self._pending_impact = {
                    "impact_frame": frame_id,
                    "impact_point": tuple(float(x) for x in best_point),
                    "person_bbox": list(person_det["bbox"]) if person_det else None,
                    "ball_bbox": list(best_ball["bbox"]),
                    "racket_bbox": list(best_racket["bbox"]),
                    "frame_height": frame_height,
                    "frame_width": frame_width,
                }

                return True 
            
        return False
     

    # ----------------------------------------------------------------------
    def process(self, frame, frame_id):



        # YOLO on full frame: person + ball (fallback when crop unusable). Racket only from RT-DETR on crop.
        detections = self.detector.detect(frame, class_filter=['person', 'ball'])
        # --- PERSON → RACKET CROP LOGIC ---
        person_det = next((d for d in detections if d['class'] == 'person'), None)
        all_rackets = []
        crop = None
        if person_det:
            x1, y1, x2, y2 = person_det['bbox']
            h, w, _ = frame.shape

            p_center = np.array([(x1+x2)/2, (y1+y2)/2])

            pad_w = (x2-x1)*0.25
            pad_h = (y2-y1)*0.25

            cx1 = int(max(0, x1-pad_w))
            cy1 = int(max(0, y1-pad_h))
            cx2 = int(min(w, x2+pad_w))
            cy2 = int(min(h, y2+pad_h))

            crop = frame[cy1:cy2, cx1:cx2]

            # RT-DETR on the person crop for ball + racket; map bboxes to full-frame coords.
            if is_valid_crop(crop):
                crop_dets = self.rtdetr.detect(crop, class_filter=['ball', 'racket'])
                crop_balls = [d for d in crop_dets if d['class'] == 'ball']
                crop_rackets = [d for d in crop_dets if d['class'] == 'racket']

                if crop_balls:
                    best_crop_ball = max(crop_balls, key=lambda b: b.get("confidence", 0.0))
                    bx1, by1, bx2, by2 = best_crop_ball["bbox"]
                    best_crop_ball["bbox"] = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
                    if "center_x" in best_crop_ball:
                        best_crop_ball["center_x"] = float(best_crop_ball["center_x"]) + cx1

                    detections = [d for d in detections if d.get("class") != "ball"]
                    detections.append(best_crop_ball)

                for r in crop_rackets:
                    bx1, by1, bx2, by2 = r['bbox']
                    r['bbox'] = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
                    all_rackets.append(r)

            wrist_xy = right_wrist_xy_full_frame(crop, cx1, cy1, fixed_height=256)
            person_max_side = max((x2 - x1), (y2 - y1))
            max_near_wrist = RACKET_TO_WRIST_MAX_FRAC * person_max_side

            best = None
            best_dist = 99999.0

            if wrist_xy is not None and all_rackets:
                wx, wy = wrist_xy
                wpt = np.array([wx, wy])
                candidates = []
                for r in all_rackets:
                    rx1, ry1, rx2, ry2 = r["bbox"]
                    rc = np.array([(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0])
                    d = float(np.linalg.norm(rc - wpt))
                    if d <= max_near_wrist:
                        candidates.append((d, r))
                if candidates:
                    candidates.sort(key=lambda t: t[0])
                    best_dist, best = candidates[0]
            else:
                # No reliable wrist: keep previous behavior (closest racket to person center)
                for r in all_rackets:
                    rx1, ry1, rx2, ry2 = r["bbox"]
                    rc = np.array([(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0])
                    d = float(np.linalg.norm(rc - p_center))
                    if d < best_dist:
                        best_dist = d
                        best = r

            max_allowed = 0.6 * person_max_side
            detections = [d for d in detections if d["class"] != "racket"]
            if best is not None:
                if wrist_xy is not None:
                    if best_dist <= max_near_wrist:
                        detections.append(best)
                elif best_dist < max_allowed:
                    detections.append(best)

        

        # REAL IMPACT
        frame_h, frame_w = frame.shape[:2]
        if crop is not None:
            flag_ = self._detect_impact_from_detections(detections, frame_id, crop, frame_h, frame_w)


        # if flag_:
        #     crop_rg = prepare_crop_for_mediapipe(crop,fixed_height=256) if crop is not None else None
        #     pose, result = infer.extract_pose(crop_rg)
        #     class_name, conf, probs = infer.predict_proba(crop_rg)
        #     print(f"class_name: {class_name}, confidence :{conf}, prob:{probs}")

        #     if result is not None:
        #         img_plt = test_plot_pose(result, crop_rg)
        #         cv2.imshow('Debug Crop', img_plt)
        #         while True:
        #             key = cv2.waitKey(0) & 0xFF
        #             if key == ord('q'):
        #                 break
        #         cv2.destroyWindow('Debug Crop')
                


        # ----------------------------------------------------------------------
        # FRAME BUFFER for shot window (impact_start .. impact_stop)
        # ----------------------------------------------------------------------
        crop_rgb = prepare_crop_for_mediapipe(crop,fixed_height=256) if crop is not None else None

        person_bbox = next((d["bbox"] for d in detections if d["class"] == "person"), None)
        ball_bbox = next((d["bbox"] for d in detections if d["class"] == "ball"), None)
        racket_bbox = next((d["bbox"] for d in detections if d["class"] == "racket"), None)
        self._frame_buffer[frame_id] = {
            "crop_rgb": crop_rgb,
            "person_bbox": list(person_bbox) if person_bbox is not None else None,
            "ball_bbox": list(ball_bbox) if ball_bbox is not None else None,
            "racket_bbox": list(racket_bbox) if racket_bbox is not None else None,
        }
        # Keep only last 11 frames
        for fid in list(self._frame_buffer.keys()):
            if fid < frame_id - 10:
                del self._frame_buffer[fid]

        # Clear overlay after linger period
        if self._display_until_frame is not None and frame_id > self._display_until_frame:
            self._last_shot = None
            self._display_until_frame = None
            self._display_impact_point = None
            self._display_impact_frame = None

        # Emit shot when we have full window after impact
        if self._pending_impact is not None:
            imp = self._pending_impact
            impact_frame = imp["impact_frame"]
            if frame_id >= impact_frame + SHOT_WINDOW_HALF:
                start_f = impact_frame - SHOT_WINDOW_HALF
                end_f = impact_frame + SHOT_WINDOW_HALF
                frame_results = []
                for fid in range(start_f, end_f + 1):
                    buf = self._frame_buffer.get(fid)
                    if buf and buf["crop_rgb"] is not None and is_valid_crop(buf["crop_rgb"]):
                        _, conf, probs = infer.predict_proba(buf["crop_rgb"])
                        frame_results.append((probs if probs else None, conf))
                    else:
                        frame_results.append((None, 0.0))
                shot_type, confidence, quality_flag = shot_analysis.aggregate_frame_predictions(frame_results)
                shot_window = (start_f, end_f)
                court_lines = detect_court_lines(frame)
                metrics = shot_analysis.compute_shot_metrics(
                    impact_frame,
                    shot_window,
                    imp["impact_point"],
                    imp["person_bbox"],
                    imp["ball_bbox"],
                    imp["racket_bbox"],
                    imp["frame_height"],
                    imp["frame_width"],
                    court_lines=court_lines,
                )
                # ------------------------------------------------------------------
                # Approximate ball speed and left/right direction after impact
                # using the tracked ball bboxes in the shot window.
                # ------------------------------------------------------------------
                ball_positions = []
                for fid in range(impact_frame, end_f + 1):
                    buf = self._frame_buffer.get(fid)
                    if not buf or buf.get("ball_bbox") is None:
                        continue
                    bx1, by1, bx2, by2 = buf["ball_bbox"]
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    ball_positions.append((fid, cx, cy))

                ball_speed_px_s = None
                direction_lr = None
                if len(ball_positions) >= 2:
                    first_f, first_x, first_y = ball_positions[0]
                    last_f, last_x, last_y = ball_positions[-1]
                    df = max(1, last_f - first_f)
                    dx = last_x - first_x
                    dy = last_y - first_y
                    # Approximate per-frame velocity, then convert to px/s (assume ~30 fps).
                    fps_est = 30.0
                    dist_per_frame = float(np.sqrt(dx**2 + dy**2) / df)
                    ball_speed_px_s = dist_per_frame * fps_est
                    # Coarse left/right direction from horizontal displacement.
                    if abs(dx) > 2.0:
                        direction_lr = "right" if dx > 0 else "left"

                metrics["ball_speed_after_impact"] = ball_speed_px_s
                metrics["ball_speed_px_s"] = ball_speed_px_s
                # Preserve any richer shot_direction from metrics if present; otherwise use coarse LR.
                if metrics.get("shot_direction") is None and direction_lr is not None:
                    metrics["shot_direction"] = direction_lr
                metrics["direction_lr"] = direction_lr

                metrics["quality_flag"] = quality_flag
                score, score_reasons, tips = shot_analysis.compute_score_and_feedback(
                    metrics, shot_type, confidence
                )
                shot_out = shot_analysis.build_shot_output(
                    shot_type, confidence, quality_flag, metrics, score, score_reasons, tips,
                    shot_window_frames=end_f - start_f + 1,
                )
                self.shots.append(shot_out)
                self._last_shot = shot_out
                self._pending_impact = None
                self.racket_ball_impact_point = None
                self.racket_ball_impact_frame = None
                # Keep overlay visible for a bit after impact
                self._display_until_frame = frame_id + OVERLAY_LINGER_FRAMES
                self._display_impact_point = tuple(int(x) for x in imp["impact_point"])
                self._display_impact_frame = impact_frame

        # ----------------------------------------------------------------------
        # DRAW (pass shot data so metrics are drawn when impact window completed)
        # ----------------------------------------------------------------------
        in_linger = self._display_until_frame is not None and frame_id <= self._display_until_frame
        draw_impact_point = self.racket_ball_impact_point or (self._display_impact_point if in_linger else None)
        draw_impact_frame = self.racket_ball_impact_frame or (self._display_impact_frame if in_linger else None)
        draw_shot_data = self._last_shot if in_linger else None
        lines = detect_court_lines(frame)
        frame = draw(
            frame, detections,
            lines,
            draw_impact_point,
            draw_impact_frame,
            shot_data=draw_shot_data,
        )
        return frame, detections, self._last_shot
