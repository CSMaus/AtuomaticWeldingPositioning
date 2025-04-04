
import os
import cv2
import numpy as np
from ultralytics import YOLO

videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/"
# this_video_path = os.path.join(videos_path, os.listdir(videos_path)[11])
# "rb6.360mm & 30d.mp4")  # "rb_test7.mp4")
video_name = "rb_test8"
this_video_path = os.path.join(videos_path, f"{video_name}.mp4")
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paused = False
frame_pos = 0

yolo_model = YOLO("runs/segment/electrode_groove_seg8/weights/best.pt")
# yolo_model = YOLO("runs/electrode_groove_seg2/weights/best.pt")

last_groove_box = None  # temporal memory

def set_frame(pos):
    global frame_pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    frame_pos = pos

def toggle_pause():
    global frame_paused
    frame_paused = not frame_paused

def nothing(x): pass

cv2.namedWindow("Filter Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Filter Controls", 400, 100)
cv2.createTrackbar("Canny Min", "Filter Controls", 98, 255, nothing)
cv2.createTrackbar("Canny Max", "Filter Controls", 173, 255, nothing)

def process_edges(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cmin = cv2.getTrackbarPos("Canny Min", "Filter Controls")
    cmax = cv2.getTrackbarPos("Canny Max", "Filter Controls")
    edges = cv2.Canny(gray, cmin, cmax)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_shifted = cnt + [x1, y1]
        cv2.drawContours(image, [cnt_shifted], -1, (0, 0, 255), 2)

def predict_yolo(curr_frame):
    global last_groove_box
    labeled = curr_frame.copy()
    results = yolo_model.predict(curr_frame, verbose=False)[0]


    names = yolo_model.names
    found_groove = False

    if results.boxes is not None:
        boxes = results.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
            cls_id = int(box.cls.item())
            label = names[cls_id]

            color = (255, 50, 150) if label == "groove center" else (50, 255, 50)
            cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label == "groove center":
                last_groove_box = xyxy
                found_groove = True

            process_edges(labeled, xyxy)

    # If groove center not detected in this frame, re-use last known position
    if not found_groove and last_groove_box is not None:
        process_edges(labeled, last_groove_box)
        cv2.rectangle(labeled, (last_groove_box[0], last_groove_box[1]), (last_groove_box[2], last_groove_box[3]), (255, 0, 0), 1)
        cv2.putText(labeled, "last groove center", (last_groove_box[0], last_groove_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return labeled

# warm-up
_ = yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

cv2.namedWindow("Welding Analysis", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", "Welding Analysis", 0, total_frames - 1, set_frame)

labeled_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        # display_frame = labeled_frame if labeled_frame is not None else frame
    # else:
    labeled_frame = predict_yolo(frame)
    display_frame = labeled_frame

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis", current_frame)
    cv2.imshow("Welding Analysis", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()
    elif key == ord("p") and frame_paused:
        labeled_frame = predict_yolo(frame)

cap.release()
cv2.destroyAllWindows()

