import os
import csv
from pathlib import Path
import cv2

root = Path.cwd().parents[2] / "data"
videos_folders_list = ["basler_recordings", "Curve_250808", "focusing data", "labeling data"]
output_path = root / "AllFrames-Data"
output_path.mkdir(parents=True, exist_ok=True)

log_csv = output_path / "saved_frames_index.csv"
if not log_csv.exists():
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame_index", "saved_file"])

def read_and_save_video_frames(video_path: Path, tosave_path: Path, frames_step: int = 5):
    name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    frame_id = 0
    saved_count = 0

    # tosave_path.mkdir(parents=True, exist_ok=True)

    with log_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frames_step == 0:
                outfile = tosave_path / f"{name}-frame_{frame_id:06d}.png"
                cv2.imwrite(str(outfile), frame)
                writer.writerow([name, frame_id, str(outfile)])
                saved_count += 1

            frame_id += 1

    cap.release()
    print(f"For video '{name}' saved {saved_count} frames (step={frames_step})")
    return saved_count

for folder in videos_folders_list:
    # sorted(os.listdir(root)):
    if folder == "old_videos":
        continue
    folder_path = root / folder
    if not folder_path.is_dir():
        continue

    # save_frames_dir = output_path / folder
    # save_frames_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in [".mp4", ".avi"]])
    for video in videos:
        read_and_save_video_frames(video, output_path, frames_step=5)
