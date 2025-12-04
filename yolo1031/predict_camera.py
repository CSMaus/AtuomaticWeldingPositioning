import cv2
from ultralytics import YOLO
import numpy as np

MODEL_PATH = "runs/segment/welding_seg_1203-/weights/best.pt"
CAM_INDEX = 0
CLASS_NAMES = {
    0: "groove center",
    1: "Electrode"
}

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run inference
    results = model(frame, imgsz=640, verbose=False)[0]

    # if segmentation model
    if results.masks is not None:
        for seg, box, cls_id in zip(results.masks.xy, results.boxes.xyxy, results.boxes.cls):
            cls_id = int(cls_id)
            if cls_id not in CLASS_NAMES:
                continue
            pts = np.round(seg).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            x1, y1, x2, y2 = box.int().tolist()
            cv2.putText(frame, CLASS_NAMES[cls_id],
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # if detection-only model (no masks)
    else:
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            cls_id = int(cls_id)
            if cls_id not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, CLASS_NAMES[cls_id],
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
