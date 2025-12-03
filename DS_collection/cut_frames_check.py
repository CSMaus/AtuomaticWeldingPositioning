# this script is to cut the video from sides to remove SW frame
# define_crop.py
import cv2

VIDEO_PATH = "original_video/20251023-1_08-1_17.mkv"
TEST_FRAME_INDEX = 15

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

cap.set(cv2.CAP_PROP_POS_FRAMES, TEST_FRAME_INDEX)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read test frame")

h, w = frame.shape[:2]

cv2.namedWindow("CROP_CAM1", cv2.WINDOW_NORMAL)
cv2.namedWindow("CROP_CAM2", cv2.WINDOW_NORMAL)

def _noop(x):
    pass

# sliders: how much to cut from each side
cv2.createTrackbar("top",    "CROP_CAM1", 0, h // 2, _noop)  # 107
cv2.createTrackbar("bottom", "CROP_CAM1", 0, h // 2, _noop)  # 191
cv2.createTrackbar("left",   "CROP_CAM1", 0, w // 2, _noop)  # 78
cv2.createTrackbar("right",  "CROP_CAM1", 0, w // 2, _noop)  # 101

while True:
    top    = cv2.getTrackbarPos("top",    "CROP_CAM1")
    bottom = cv2.getTrackbarPos("bottom", "CROP_CAM1")
    left   = cv2.getTrackbarPos("left",   "CROP_CAM1")
    right  = cv2.getTrackbarPos("right",  "CROP_CAM1")

    y1, y2 = top,   h - bottom
    x1, x2 = left,  w - right

    if y2 <= y1 or x2 <= x1:
        roi = frame
    else:
        roi = frame[y1:y2, x1:x2]

    # split ROI into 2 cameras (left/right)
    mid = roi.shape[1] // 2
    cam1 = roi[:, :mid]
    cam2 = roi[:, mid:]

    cv2.imshow("CROP_CAM1", cam1)
    cv2.imshow("CROP_CAM2", cam2)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("preview_cam1.png", cam1)
        cv2.imwrite("preview_cam2.png", cam2)
        print(f"top={top}, bottom={bottom}, left={left}, right={right}")

cap.release()
cv2.destroyAllWindows()
