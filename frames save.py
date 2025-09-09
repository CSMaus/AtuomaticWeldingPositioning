# here will save each 5th frame
import cv2
import os

# videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/basler_recordings/"
save_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/br_frames-0909/"
video_name = os.listdir(videos_path)[5]
this_video_path = os.path.join(videos_path, video_name)
print("Output path: ", save_path)
# f"dataset/frames/{video_name[:-4]}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(this_video_path)

count = 0
saved_num = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    cv2.imshow("video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"{save_path}/{video_name[:-4]}-frame_{count:04d}.png", frame)
        print("Saved to: ", f"{save_path}/{video_name[:-4]}-frame_{count:04d}.png")
        saved_num += 1

    elif key == ord('q'):
        break
    count += 1

print(f"saved {saved_num} frames")
cap.release()
cv2.destroyAllWindows()
