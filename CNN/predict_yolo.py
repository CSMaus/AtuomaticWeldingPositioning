# open wideo and save selected frames by pressing "s"
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


def set_frame(pos): global frame_pos; cap.set(cv2.CAP_PROP_POS_FRAMES, pos); frame_pos = pos
def toggle_pause(): global frame_paused; frame_paused = not frame_paused


def predict_yoloaa(curr_frame):
    results = yolo_model.predict(curr_frame, conf=0.0, verbose=False)  # force prediction
    labeled = curr_frame.copy()

    if results and results[0].boxes:
        names = yolo_model.names
        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            cls_id = int(box.cls.item())
            label = names[cls_id]
            cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return labeled


def predict_yolo(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()

    names = yolo_model.names
    if results.boxes is not None:
        color_electrode = (50, 255, 50)
        color_groove = (255, 50, 150)
        did_el_bbox = False
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
            cls_id = int(box.cls.item())
            label = names[cls_id]
            if did_el_bbox:
                cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color_electrode, 2)
            else:
                cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color_groove, 2)
            cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            did_el_bbox = True

    return labeled


# Warm-up YOLO with a dummy image
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
        display_frame = labeled_frame if labeled_frame is not None else frame
    else:
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


