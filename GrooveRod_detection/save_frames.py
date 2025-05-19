import sys

import cv2
import os
from pathlib import Path

path = os.path.join(Path.cwd().parents[2], "data/")

folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
output_path = os.path.join(path, "RL_Groove_Rod-Data/")

if not os.path.exists(output_path):
    os.makedirs(output_path)

def read_and_save_video_frames(video_path, tosave_path, frames_step = 5):
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    # save_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frames_step == 0:
            cv2.imwrite(f"{tosave_path}/{name}-frame_{frame_id:04d}.jpg", frame)
            # save_id += 1
        frame_id += 1

    print(f"For video '{name}' saved {frame_id // 5 + 1} frames")
    cap.release()


# TODO: I think we can use same annotation for same frame index for same folder, bcs they are different only with the focusing.
# TODO: test these annotations **WITH adjusted for initial frame shift**
for folder in folders:
    if folder != "old_videos":
        save_frames = os.path.join(output_path, folder)
        if not os.path.exists(save_frames):
            os.makedirs(save_frames)

        videos_path = os.path.join(path, folder)
        videos = [f for f in os.listdir(videos_path) if f.endswith(".mp4") and os.path.isfile(os.path.join(videos_path, f))]
        for video in videos:
            read_and_save_video_frames(os.path.join(videos_path, video), save_frames)




