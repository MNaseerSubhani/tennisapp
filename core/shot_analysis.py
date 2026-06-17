"""
Shot analysis: aggregate predictions over impact window, compute metrics,
rule-based scoring, and actionable tips. Output conforms to a stable JSON schema.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import core.infer as infer

# -----------------------------------------------------------------------------
# Aggregation: average predictions over frames [impact_start .. impact_stop]
# -----------------------------------------------------------------------------
def aggregate_frame_predictions(
    frame_results: List[Tuple[Optional[Dict[str, float]], float]]
) -> Tuple[str, float, str]:
    valid = [(p, c) for p, c in frame_results if p is not None and isinstance(p, dict) and len(p) > 0]
    if not valid:
        return (infer.MACRO_UNKNOWN, 0.0, "low")

    probs_list = [p for p, _ in valid]
    confidences = [c for _, c in valid]

    # Average probability per class
    all_classes = set()
    for p in probs_list:
        all_classes.update(p.keys())
    mean_probs = {}
    for cls in all_classes:
        vals = [p.get(cls, 0.0) for p in probs_list]
        mean_probs[cls] = float(np.mean(vals))

    if not mean_probs:
        return (infer.MACRO_UNKNOWN, 0.0, "low")

    # Class = argmax over mean probabilities (unchanged)
    sorted_classes = sorted(mean_probs.items(), key=lambda x: x[1], reverse=True)
    best_class, best_prob = sorted_classes[0]

    # Confidence from entropy of mean_probs (works well for small datasets;
    # 0.4–0.5 max prob can still yield good confidence when distribution is peaked)
    probs_arr = np.array(list(mean_probs.values()), dtype=float)
    n_classes = len(probs_arr)
    eps = 1e-10
    entropy = -float(np.sum(probs_arr * np.log(probs_arr + eps)))
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    mean_confidence = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))

    # If overall confidence is very low, treat as unknown shot
    if mean_confidence < 0.2:
        best_class = infer.MACRO_UNKNOWN
        quality_flag = "low"
    else:
        # Quality flag from entropy-based confidence and number of frames
        n_valid = len(valid)
        if mean_confidence >= 0.7 and n_valid >= 5:
            quality_flag = "high"
        elif mean_confidence >= 0.5 and n_valid >= 3:
            quality_flag = "medium"
        else:
            quality_flag = "low"

    return (best_class, mean_confidence, quality_flag)
# -----------------------------------------------------------------------------
# Metrics from impact context (frame, bboxes, court, optional tracking)
# -----------------------------------------------------------------------------

def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_shot_metrics(
    impact_frame: int,
    shot_window: Tuple[int, int],
    impact_point_xy: Tuple[float, float],
    person_bbox: Optional[List[float]],
    ball_bbox: Optional[List[float]],
    racket_bbox: Optional[List[float]],
    frame_height: int,
    frame_width: int,
    fps: float = 30.0,
    ball_speed_after_impact: Optional[float] = None,
    ball_velocity_xy: Optional[Tuple[float, float]] = None,
    court_lines: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Build metrics dict for one shot. All numeric metrics in consistent units where possible.
    """
    ix, iy = impact_point_xy
    metrics = {
        "impact_time": impact_frame,
        "shot_window": {"start": shot_window[0], "end": shot_window[1]},
        "impact_height": None,
        "impact_distance_to_body": None,
        "impact_offset_forward": None,
        "player_zone": None,
        "ball_speed_after_impact": ball_speed_after_impact,
        "shot_direction": None,
        "quality_flag": "medium",
    }

    # Impact height: normalized 0 (top) .. 1 (bottom) in image
    if frame_height > 0:
        metrics["impact_height"] = float(np.clip(iy / frame_height, 0.0, 1.0))

    if person_bbox is not None:
        px, py = _bbox_center(person_bbox)
        metrics["impact_distance_to_body"] = float(
            np.sqrt((ix - px) ** 2 + (iy - py) ** 2)
        )
        # Proxy for early/optimal/late: positive = impact in front of body (right of center in image)
        metrics["impact_offset_forward"] = float(ix - px)

    # Player zone: coarse from image x (left/center/right third) if no court lines
    if frame_width > 0:
        third = frame_width / 3
        if ix < third:
            metrics["player_zone"] = "left"
        elif ix > 2 * third:
            metrics["player_zone"] = "right"
        else:
            metrics["player_zone"] = "center"
    if court_lines:
        metrics["player_zone"] = metrics["player_zone"] or "center"

    # Shot direction from ball velocity after impact (if tracking)
    if ball_velocity_xy is not None:
        vx, vy = ball_velocity_xy
        if abs(vx) + abs(vy) > 1e-3:
            # Cardinal proxy: left/right from vx, up/down from vy
            h = "right" if vx > 0 else "left"
            v = "up" if vy < 0 else "down"
            metrics["shot_direction"] = f"{v}_{h}"

    return metrics


def compute_score_and_feedback(
    metrics: Dict[str, Any],
    shot_type: str,
    confidence: float,
) -> Tuple[int, List[str], List[str]]:
    """
    Rule-based score 0-100, explainable score_reasons, and 2-3 actionable tips.
    Returns (score, score_reasons, tips).
    The score is explicitly tied to classification confidence so that
    uncertain shots cannot receive very high scores.
    """
    # Start from a neutral base; other metrics and confidence will move this.
    score = 60
    reasons = []
    tips = []

    # Impact height: ideal contact often around mid-body in image (0.4-0.6)
    h = metrics.get("impact_height")
    if h is not None:
        if 0.35 <= h <= 0.65:
            score += 10
            reasons.append("impact_height_in_ideal_range")
        elif h < 0.25 or h > 0.75:
            score -= 15
            reasons.append("impact_height_too_extreme")
            tips.append("Try to make contact at waist-to-chest height for better control.")

    # Distance to body: not too close (cramped) or too far (stretched)
    d = metrics.get("impact_distance_to_body")
    if d is not None and d > 0:
        # Heuristic: assume ~100-300 px is "comfortable" (depends on resolution)
        if 80 <= d <= 350:
            score += 5
            reasons.append("comfortable_contact_distance")
        elif d > 450:
            score -= 10
            reasons.append("contact_too_far_from_body")
            tips.append("Move your feet to get the ball in front of your body at contact.")

    # Offset forward: slight forward is good (contact in front)
    off = metrics.get("impact_offset_forward")
    if off is not None:
        if 20 <= off <= 150:
            score += 5
            reasons.append("contact_slightly_in_front")
        elif off < -50:
            score -= 10
            reasons.append("contact_behind_body")
            tips.append("Hit the ball slightly in front of your body for more power and accuracy.")

    # Classification confidence – strongly tie score to certainty
    from core import infer

    if shot_type == infer.MACRO_UNKNOWN or confidence < infer.CONFIDENCE_THRESHOLD:
        # Low-confidence / unknown predictions should never look like top-quality shots.
        score -= 15
        reasons.append("low_or_unknown_classification_confidence")
        # Cap maximum score for unknown/low-confidence shots.
        score = min(score, 55)
        if "The system was not fully confident in the shot type." not in tips:
            tips.append("The system was not fully confident in the shot type; focus on clear preparation and contact.")
    else:
        if confidence >= 0.85:
            score += 12
            reasons.append("very_high_classification_confidence")
        elif confidence >= 0.75:
            score += 8
            reasons.append("high_classification_confidence")
        elif confidence >= 0.65:
            score += 3
            reasons.append("moderate_classification_confidence")
        else:
            # Slight penalty for very borderline but above-threshold cases
            score -= 3
            reasons.append("borderline_classification_confidence")

    # Cap and ensure at least 2-3 tips
    score = int(np.clip(score, 0, 100))

    if not tips:
        tips.append("Keep a compact swing through the contact zone.")
        tips.append("Recover to ready position after each shot.")
    if len(tips) < 2:
        tips.append("Watch the ball until contact.")

    return (score, reasons, tips[:3])


# -----------------------------------------------------------------------------
# Full shot output: stable JSON schema for backend/frontend
# -----------------------------------------------------------------------------

def build_shot_output(
    shot_type: str,
    confidence: float,
    quality_flag: str,
    metrics: Dict[str, Any],
    score: int,
    score_reasons: List[str],
    tips: List[str],
    shot_window_frames: int,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Single shot result conforming to the documented schema.
    """
    impact_frame = metrics.get("impact_time")
    shot_window = metrics.get("shot_window", {})
    start, end = shot_window.get("start", 0), shot_window.get("end", 0)
    return {
        "shot_type": shot_type,
        "confidence": round(confidence, 4),
        "quality_flag": quality_flag,
        "metrics": {
            "impact_time": impact_frame,
            "shot_window": shot_window,
            "impact_height": metrics.get("impact_height"),
            "impact_distance_to_body": metrics.get("impact_distance_to_body"),
            "impact_offset_forward": metrics.get("impact_offset_forward"),
            "player_zone": metrics.get("player_zone"),
            # Backwards-compatible naming plus explicit px/s + left/right.
            "ball_speed_after_impact": metrics.get("ball_speed_after_impact"),
            "ball_speed_px_s": metrics.get("ball_speed_px_s"),
            "shot_direction": metrics.get("shot_direction"),
            "direction_lr": metrics.get("direction_lr"),
            "confidence_values": {"classification": round(confidence, 4)},
            "quality_flag": quality_flag,
        },
        "scoring": {
            "score": score,
            "score_reasons": score_reasons,
            "tips": tips,
        },
    }


# -----------------------------------------------------------------------------
# Documented JSON schema (for backend/frontend integration)
# -----------------------------------------------------------------------------

SHOT_SCHEMA = {
    "description": "Per-shot analysis result. Stable schema for API/UI.",
    "type": "object",
    "properties": {
        "shot_type": {"type": "string", "description": "Class name or 'unknown' if low confidence"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "quality_flag": {"type": "string", "enum": ["high", "medium", "low"]},
        "metrics": {
            "type": "object",
            "properties": {
                "impact_time": {"type": "integer"},
                "shot_window": {"type": "object", "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}}},
                "impact_height": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "impact_distance_to_body": {"type": ["number", "null"]},
                "impact_offset_forward": {"type": ["number", "null"]},
                "player_zone": {"type": ["string", "null"]},
                "ball_speed_after_impact": {"type": ["number", "null"]},
                "ball_speed_px_s": {"type": ["number", "null"], "description": "Approximate post-impact ball speed in pixels/second."},
                "shot_direction": {"type": ["string", "null"]},
                "direction_lr": {"type": ["string", "null"], "description": "Coarse left/right direction after impact."},
                "confidence_values": {"type": "object"},
                "quality_flag": {"type": "string"},
            },
        },
        "scoring": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
                "score_reasons": {"type": "array", "items": {"type": "string"}},
                "tips": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
            },
        },
    },
    "required": ["shot_type", "confidence", "quality_flag", "metrics", "scoring"],
}
