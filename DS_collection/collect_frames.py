# this script is to save frames pressing s button
# for press s will be saved 2 frames -: cam1 and cam2

import cv2
import os
import numpy as np

VIDEO_PATH = "original_video/20251023-1_08-1_17.mkv"
OUTPUT_DIR = "saved_frames/" + VIDEO_PATH.replace("original_video/", "").replace(".mkv", "")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP    = 107
BOTTOM = 191
LEFT   = 78
RIGHT  = 101

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("Video_2cams", cv2.WINDOW_NORMAL)

current_frame = 0
paused = False
last_cam1 = None
last_cam2 = None

def crop_and_split(frame):
    h, w = frame.shape[:2]
    y1, y2 = TOP,    h - BOTTOM
    x1, x2 = LEFT,   w - RIGHT

    if y2 <= y1 or x2 <= x1:
        roi = frame
    else:
        roi = frame[y1:y2, x1:x2]

    mid = roi.shape[1] // 2
    cam1 = roi[:, :mid]
    cam2 = roi[:, mid:]
    return cam1, cam2

def on_trackbar(pos):
    global current_frame, paused, last_cam1, last_cam2
    current_frame = pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        cam1, cam2 = crop_and_split(frame)
        last_cam1, last_cam2 = cam1, cam2
        combined = np.hstack((cam1, cam2))
        cv2.imshow("Video_2cams", combined)

cv2.createTrackbar("frame", "Video_2cams", 0, max(total_frames - 1, 1), on_trackbar)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        cam1, cam2 = crop_and_split(frame)
        last_cam1, last_cam2 = cam1, cam2
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        cv2.setTrackbarPos("frame", "Video_2cams", current_frame)
        combined = np.hstack((cam1, cam2))
        cv2.imshow("Video_2cams", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    elif key == ord(' '):
        paused = not paused
    elif key == ord('s') and last_cam1 is not None and last_cam2 is not None:
        base = f"frame_{current_frame:08d}"
        path1 = os.path.join(OUTPUT_DIR, base + "_cam1.png")
        path2 = os.path.join(OUTPUT_DIR, base + "_cam2.png")
        cv2.imwrite(path1, last_cam1)
        cv2.imwrite(path2, last_cam2)
        print(f"Saved {path1} and {path2}")

cap.release()
cv2.destroyAllWindows()
