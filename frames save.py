# here will save each 5th frame
import cv2
import os
import glob

# videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
# videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/basler_recordings/"
# save_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/br_frames-0909/"
videos_path = "data/Curve_250808/"
save_path = "data/Curve_250808-0911/"
files = glob.glob(os.path.join(videos_path, '*.mp4'))

this_video_path = files[6]  # os.listdir(videos_path)[5]; this_video_path = os.path.join(videos_path, video_name)
video_name = this_video_path.split("\\")[-1]
print("Video name       :", video_name)

print("Opened video path: ", this_video_path)
print("Output path      : ", save_path)
# f"dataset/frames/{video_name[:-4]}"


if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(this_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

count = 0
saved_num = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    cv2.imshow("video", frame)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"{save_path}/{video_name[:-4]}-frame_{count:04d}.png", frame)
        print("Saved to: ", f"{save_path}/{video_name[:-4]}-frame_{count:04d}.png")
        saved_num += 1

    elif key in (27, ord('q')): break
    count += 1

print(f"saved {saved_num} frames")
cap.release()
cv2.destroyAllWindows()
