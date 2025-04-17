# here is the script without windowed GUI
# all parameters could be adjusted in the script
from helpers import predict_yolo45, get_masks_points_distance45, draw_masks_points_distance  # , predict_yolo
import numpy as np
from ultralytics import YOLO
import cv2

yolo_model = YOLO("electrode_groove_seg45/weights/best.pt") # was trained on frames rotated 45 degrees
# yolo_model = YOLO("electrode_groove_seg45/weights/best.pt")
# warm-up the model to not fave freeze in first frame with dummy image
_ = yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
electrode_width = 4.03  # mm
camera_rotation_angle = 0


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # json_data = get_masks_points_distance(frame, electrode_width, yolo_model, camera_rotation_angle)
    json_data = get_masks_points_distance45(frame, electrode_width, yolo_model, camera_rotation_angle)
    labeled_frame = draw_masks_points_distance(frame, json_data)

    cv2.imshow("Camera", labeled_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()