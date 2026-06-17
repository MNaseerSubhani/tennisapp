

import cv2
import numpy as np

COLORS = {
    "ball": (0, 0, 255),
    "person": (0, 255, 0),
    "racket": (255, 0, 0)
    
    
}

def draw(frame, detections, court_lines, racket_ball_impact_point, racket_ball_impact_frame, shot_data=None):
    # Keep a clean copy of the original frame (video content)
    video_frame = frame.copy()

    # Draw detections (no track filter when tracking is disabled)
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = COLORS[d["class"]]
        cv2.rectangle(video_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(video_frame, d["class"], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # # Draw tracks
    # for t in tracks:
    #     if t["track_id"] not in selected_ids.values():
    #         continue
    #     x1, y1, x2, y2 = map(int, t["bbox"])
    #     cv2.putText(frame, f'ID {t["track_id"]}', (x1, y2 + 15),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # # Draw current positions
    # for p in trajectories:
    #     if p["track_id"] not in selected_ids.values():
    #         continue
    #     x, y = map(int, p["position"])
    #     cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

    # Draw trajectory histories as lines
    # for track_id, history in trajectory_histories.items():
    #     if track_id not in selected_ids.values() or len(history) < 2:
    #         continue
    #     points = np.array(history, dtype=np.int32)
    #     color = COLORS[next((d["class"] for d in detections if "track_id" in d and d["track_id"] == track_id), "person")]
    #     cv2.polylines(frame, [points], False, color, 2)

    # Draw court lines on the video only
    for l in court_lines:
        cv2.line(video_frame, tuple(l["start"]), tuple(l["end"]), (255, 255, 0), 2)
    
    # if min_ is not None:
    #     cv2.putText(frame, f"min dis: {min_}", (60, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Draw racket-ball impact point on the video
    if racket_ball_impact_point is not None and racket_ball_impact_frame is not None:
        cv2.circle(video_frame, racket_ball_impact_point, 10, (0, 255, 255), -1)  # Yellow circle for impact
        # cv2.putText(frame, f"Impact detected at frame: {racket_ball_impact_frame}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Create a separate HUD panel (same size as video) and draw all metrics there
    hud_panel = np.zeros_like(video_frame)

    # Draw all metrics on HUD when impact window has been processed (shot_data present)
    if shot_data is not None:
        metrics = shot_data.get("metrics") or {}
        scoring = shot_data.get("scoring") or {}

        # Layout for the on-screen HUD
        h, w = hud_panel.shape[:2]
        x_start = 20
        y_offset = 60
        line_height = 26
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        # Helper to draw text with a solid background box for readability
        def draw_text_with_bg(text, x, y, font, font_scale, text_color, bg_color, thickness=1, padding=4):
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x2 = min(w - 10, x + tw + 2 * padding)
            y2 = min(h - 10, y + padding)
            y1 = max(10, y - th - padding)
            cv2.rectangle(hud_panel, (x, y1), (x2, y2), bg_color, -1)
            cv2.putText(hud_panel, text, (x + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        header_text = f"Shot: {shot_data.get('shot_type', '?')} (conf: {shot_data.get('confidence', 0):.2f})"
        draw_text_with_bg(header_text, x_start, y_offset, font, font_scale, (0, 255, 255), bg_color, thickness=1)
        y_offset += line_height

        for key, value in metrics.items():
            if value is None or isinstance(value, dict):
                text = f"  {key}: {value}"
            else:
                text = f"  {key}: {value}"
            draw_text_with_bg(text, x_start, y_offset, font, font_scale, color, bg_color, thickness=1)
            y_offset += line_height

        draw_text_with_bg(f"Score: {scoring.get('score', '?')}", x_start, y_offset, font, font_scale, (0, 255, 0), bg_color, thickness=1)
        y_offset += line_height

        for reason in scoring.get("score_reasons", [])[:3]:
            draw_text_with_bg(f"  - {reason}", x_start, y_offset, font, font_scale, color, bg_color, thickness=1)
            y_offset += line_height

        for tip in scoring.get("tips", [])[:3]:
            draw_text_with_bg(f"  Tip: {tip}", x_start, y_offset, font, font_scale, (200, 200, 100), bg_color, thickness=1)
            y_offset += line_height

    # Stack video and HUD side-by-side so text is not overlaid on the video itself
    combined = np.hstack((video_frame, hud_panel))
    return combined

