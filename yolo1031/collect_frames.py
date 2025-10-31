# save frames from videos which I already saved and annotated using cvat
# not for yolo train DS preparation
import os, csv, cv2
import sys
from pathlib import Path

root = Path.cwd().parents[1] / "data"  # [1] for oem pc, [2] for my other
# print(root)
# sys.exit()
videos_folders_list = ["basler_recordings", "Curve_250808", "focusing data", "labeling data"]
output_path = root / "AllFrames-Data"
output_path.mkdir(parents=True, exist_ok=True)

log_csv = output_path / "saved_frames_index.csv"
if not log_csv.exists():
    raise FileNotFoundError(str(log_csv))

targets = {}
with log_csv.open("r", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        v = row["video"]
        i = int(row["frame_index"])
        if v not in targets:
            targets[v] = set()
        targets[v].add(i)

def read_and_save_frames_from_list(video_path: Path, tosave_path: Path, indices):
    name = video_path.stem
    if name not in targets:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0
    wanted = sorted(targets[name])
    saved_count = 0
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        outfile = tosave_path / f"{name}-frame_{idx:06d}.png"
        cv2.imwrite(str(outfile), frame)
        saved_count += 1
    cap.release()
    print(f"For video '{name}' saved {saved_count} frames")
    return saved_count

for folder in videos_folders_list:
    if folder == "old_videos":
        continue
    folder_path = root / folder
    if not folder_path.is_dir():
        continue
    videos = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in [".mp4", ".avi"]])
    for video in videos:
        read_and_save_frames_from_list(video, output_path, targets.get(video.stem, set()))
