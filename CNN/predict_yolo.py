# open wideo and save selected frames by pressing "s"
import os
import cv2
from prediction_functions import *

# videos_path = "/Users/kseni/Downloads/kakao/Robot REC/"
videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/"
# this_video_path = os.path.join(videos_path, os.listdir(videos_path)[11])
# "rb6.360mm & 30d.mp4")  # "rb_test7.mp4")
video_name = "rb_test6"
this_video_path = os.path.join(videos_path, f"{video_name}.mp4")
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paused = False
frame_pos = 0
electrode_width = 4.03  # mm

# yolo_model = YOLO("runs/segment/electrode_groove_seg45/weights/best.pt")
# yolo_model = YOLO("runs/electrode_groove_seg2/weights/best.pt")


def set_frame(pos): global frame_pos; cap.set(cv2.CAP_PROP_POS_FRAMES, pos); frame_pos = pos
def toggle_pause(): global frame_paused; frame_paused = not frame_paused


# warm-up with dummy image
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
        # rotate 45 -> predict -> rotate back
        labeled_frame = predict_yolo45(frame, electrode_width, False)
        display_frame = labeled_frame
        # predict_deeplab(frame, show_fps=True)  # test deeplab, but it's too slow
        # display_frame = frame

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis", current_frame)
    cv2.imshow("Welding Analysis", display_frame)



    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()
    elif key == ord("p") and frame_paused:
        labeled_frame = predict_yolo(frame, electrode_width, False)





cap.release()
cv2.destroyAllWindows()


