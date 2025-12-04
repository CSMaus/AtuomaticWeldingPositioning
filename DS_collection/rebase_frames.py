import os
import shutil

BASE_DIR = "saved_frames/20251023-1_08-1_17"

CAM1_DIR = os.path.join(BASE_DIR, "cam1")
CAM2_DIR = os.path.join(BASE_DIR, "cam2")

os.makedirs(CAM1_DIR, exist_ok=True)
os.makedirs(CAM2_DIR, exist_ok=True)

for fname in os.listdir(BASE_DIR):
    if fname.endswith("_cam1.png"):
        shutil.move(
            os.path.join(BASE_DIR, fname),
            os.path.join(CAM1_DIR, fname)
        )
    elif fname.endswith("_cam2.png"):
        shutil.move(
            os.path.join(BASE_DIR, fname),
            os.path.join(CAM2_DIR, fname)
        )

print("O")
