# Shot analysis output schema

Stable JSON schema for backend/frontend integration. Each shot is emitted when the impact window is complete (frames `impact_time - 5` to `impact_time + 5`).

## Per-shot object

| Field | Type | Description |
|-------|------|-------------|
| `shot_type` | string | Class name from pose classifier, or `"unknown"` when confidence is low |
| `confidence` | number (0–1) | Classification confidence; low → `shot_type` may be `"unknown"` |
| `quality_flag` | `"high"` \| `"medium"` \| `"low"` | From confidence and number of valid prediction frames |
| `metrics` | object | See below |
| `scoring` | object | Rule-based score, reasons, and tips |

### `metrics`

| Key | Type | Description |
|-----|------|-------------|
| `impact_time` | integer | Frame index of impact |
| `shot_window` | `{ start, end }` | Frames used for prediction (impact ± 5) |
| `impact_height` | number \| null | Normalized 0–1 (top to bottom in image) |
| `impact_distance_to_body` | number \| null | Pixels from impact to person bbox center |
| `impact_offset_forward` | number \| null | Early/optimal/late proxy (pixels; positive = in front of body) |
| `player_zone` | string \| null | `"left"` \| `"center"` \| `"right"` in image |
| `ball_speed_after_impact` | number \| null | When tracking is available |
| `shot_direction` | string \| null | e.g. `"up_right"` when tracking allows |
| `confidence_values` | object | At least `classification` (0–1) |
| `quality_flag` | string | Same as top-level |

### `scoring`

| Key | Type | Description |
|-----|------|-------------|
| `score` | integer (0–100) | Rule-based, explainable |
| `score_reasons` | string[] | Links to metrics (e.g. `impact_height_in_ideal_range`) |
| `tips` | string[] | 2–3 actionable tips (no generic advice) |

## Files

- **`output/results.json`**: Array of shot objects (one per detected impact). This is the main JSON containing all impact data (metrics, scoring, etc.).

## Example

```json
{
  "shot_type": "Volea",
  "confidence": 0.72,
  "quality_flag": "high",
  "metrics": {
    "impact_time": 42,
    "shot_window": { "start": 37, "end": 47 },
    "impact_height": 0.48,
    "impact_distance_to_body": 120.5,
    "impact_offset_forward": 45.0,
    "player_zone": "center",
    "ball_speed_after_impact": null,
    "shot_direction": null,
    "confidence_values": { "classification": 0.72 },
    "quality_flag": "high"
  },
  "scoring": {
    "score": 85,
    "score_reasons": ["impact_height_in_ideal_range", "comfortable_contact_distance", "high_classification_confidence"],
    "tips": ["Keep a compact swing through the contact zone.", "Recover to ready position after each shot."]
  }
}
```
