# open wideo and save selected frames by pressing "s"
import os
import cv2

videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/"
# this_video_path = os.path.join(videos_path, os.listdir(videos_path)[11])
# "rb6.360mm & 30d.mp4")  # "rb_test7.mp4")
video_name = "rb_test6"
this_video_path = os.path.join(videos_path, f"{video_name}.mp4")
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
to_save_path = "ds_imgs/"
frame_paused = False
frame_pos = 0


def set_frame(pos): global frame_pos; cap.set(cv2.CAP_PROP_POS_FRAMES, pos); frame_pos = pos
def toggle_pause(): global frame_paused; frame_paused = not frame_paused
def save_current_frame(cframe, frame_idx):
    path_to_save = os.path.join(to_save_path, f"{video_name}_{frame_idx}.png")
    cv2.imwrite(path_to_save, cframe); print("Saved frame: ", frame_idx)


cv2.namedWindow("Welding Analysis", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", "Welding Analysis", 0, total_frames - 1, set_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis", current_frame)
    cv2.imshow("Welding Analysis", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()
    elif key == ord("s"):
        save_current_frame(frame, current_frame)


