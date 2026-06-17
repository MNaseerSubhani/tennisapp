import cv2
from core.pipeline import VisionPipeline
from storage.json_writer import save
from pathlib import Path
import os



def process_video(video_path, output_path="output"):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Output frame is video + HUD panel side by side → double width
    out_width = frame_width * 2
    out_height = frame_height
    out = cv2.VideoWriter(
        os.path.join(output_path, 'output.mp4'),
        fourcc,
        fps,
        (out_width, out_height)
    )
    pipeline = VisionPipeline()
    results = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, d, last_shot = pipeline.process(frame, frame_id)
        results.append({
            "frame_id": frame_id,
            "detections": d,
        })
        frame_id += 1
        out.write(processed_frame)
    cap.release()
    out.release()

    # json_file = Path(output_path) / "results.json"
    # save(results, json_file)
    # All shots/impact data (metrics, scoring, etc.) in one main JSON file
    impacts_file = Path(output_path) / "results.json"
    save(pipeline.shots, impacts_file)
    print(f"Processing complete. Output video saved to {output_path}, {len(pipeline.shots)} impacts to {impacts_file}")

if __name__ == "__main__":
    input_video_path = "test/vid2.mp4"
    process_video(input_video_path)

