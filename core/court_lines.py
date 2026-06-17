
import cv2
import numpy as np

def detect_court_lines(frame):
    # 1. Convert to HSV (best for color filtering)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 2. White color range (court lines) - tightened for precision
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # 3. Clean mask with morphology
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    # 4. Edge detection ONLY on white mask
    edges = cv2.Canny(white_mask, 75, 200)
    # 5. Hough transform with refined parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=150,
        minLineLength=300,
        maxLineGap=20
    )
    output = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Angle filtering for horizontal/vertical lines
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > 4 * dy or dy > 4 * dx:  # Slightly relaxed for better detection
                output.append({
                    "start": [int(x1), int(y1)],
                    "end": [int(x2), int(y2)]
                })
    return output

