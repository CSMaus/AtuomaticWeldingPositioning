# here will save each 5th frame
import cv2
import os

videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
video_name = os.listdir(videos_path)[11]
this_video_path = os.path.join(videos_path, video_name)
output_path = f"dataset/frames/{video_name[:-4]}"

if not os.path.exists(output_path):
    os.makedirs(output_path)

cap = cv2.VideoCapture(this_video_path)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % 5 == 0:
        cv2.imwrite(f"{output_path}/frame_{frame_id:04d}.jpg", frame)
    frame_id += 1

print(f"Saved {frame_id // 5 + 1} frames.")
cap.release()
