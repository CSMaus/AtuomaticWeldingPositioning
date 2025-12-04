# this is for yolo12 and recently collected 3.5k images...
import os
import sys

import cv2
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import random
from pathlib import Path

CLASS_MAP = {
    "groove center": 0,
    "Electrode": 1
}

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = Path.cwd().parents[0] / "data"
images_dir1 = os.path.join(data_dir, "20251023-1_08-1_17_cam1")
images_dir2 = os.path.join(data_dir, "20251023-1_08-1_17_cam2")

image_dirs = [images_dir1, images_dir2]

print("Images dir 1:", images_dir1)
print(len(os.listdir(os.path.join(images_dir1))))
annotations_dir = os.path.join(data_dir)
print("annotations_dir:", annotations_dir)
print(os.listdir(os.path.join(annotations_dir)))
output_dir = os.path.join(script_dir, "dataset")
print("output_dir", output_dir)

image_dir_names = [os.path.basename(d) for d in image_dirs]
wanted_xml_files = {f"annotations-{name}.xml" for name in image_dir_names}

# sys.exit()

def parse_xml_dir(ann_dir, allowed_files=None):
    annotations = {}
    for p in sorted(Path(ann_dir).iterdir()):
        # if p.is_file() and p.suffix.lower() == ".xml" and p.name.startswith("annotations-"):
        if not p.is_file() or p.suffix.lower() != ".xml":
            continue
        if allowed_files is not None and p.name not in allowed_files:
            continue
        if not p.name.startswith("annotations-"):
            continue
        tree = ET.parse(str(p))
        root = tree.getroot()
        for image in root.findall(".//image"):
            name = os.path.basename(image.get("name"))
            width = int(image.get("width"))
            height = int(image.get("height"))
            polys = []
            for polygon in image.findall("polygon"):
                points_str = polygon.get("points")
                label = polygon.get("label", "")
                pts = [tuple(map(float, s.split(","))) for s in points_str.strip().split(";")]
                polys.append({"points": pts, "label": label})
            if name not in annotations:
                annotations[name] = {"width": width, "height": height, "polygons": []}
            annotations[name]["polygons"].extend(polys)
    return annotations

def normalize_points(points, width, height):
    return [(x / width, y / height) for x, y in points]

def save_yolo_label(label_path, polygons, width, height):
    with open(label_path, "w") as f:
        for polygon in polygons:
            label = polygon["label"]
            class_id = CLASS_MAP.get(label, None)
            if class_id is None:
                continue
            norm_points = normalize_points(polygon["points"], width, height)
            flat_points = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
            line = f"{class_id} " + " ".join(flat_points) + "\n"
            f.write(line)

def prepare_yolo_dataset():
    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # ann = parse_xml_dir(annotations_dir)
    ann = parse_xml_dir(annotations_dir, allowed_files=wanted_xml_files)

    all_items = []
    for img_name, ann_data in ann.items():
        img_path = None

        # try exact name in both folders
        for img_dir in image_dirs:
            candidate = os.path.join(img_dir, img_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            base, _ = os.path.splitext(os.path.basename(img_name))
            for img_dir in image_dirs:
                for ext in (".png", ".jpg", ".jpeg"):
                    alt = os.path.join(img_dir, base + ext)
                    if os.path.exists(alt):
                        img_path = alt
                        img_name = base + ext
                        break
                if img_path is not None:
                    break

        if img_path is None:
            continue

        polys = [p for p in ann_data["polygons"] if (p.get("label") in CLASS_MAP)]
        if not any(p.get("label") == "groove center" for p in polys):
            continue
        h, w = ann_data.get("height"), ann_data.get("width")
        if not (w and h):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
        all_items.append((img_path, {"width": w, "height": h, "polygons": polys}, img_name))
    random.shuffle(all_items)
    split_idx = int(0.8 * len(all_items))
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    for img_path, ann_data, img_name in tqdm(train_items, desc="train"):
        dst_img = f"{output_dir}/images/train/{img_name}"
        shutil.copy2(img_path, dst_img)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = f"{output_dir}/labels/train/{label_name}"
        save_yolo_label(label_path, ann_data["polygons"], ann_data["width"], ann_data["height"])
    for img_path, ann_data, img_name in tqdm(val_items, desc="val"):
        dst_img = f"{output_dir}/images/val/{img_name}"
        shutil.copy2(img_path, dst_img)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = f"{output_dir}/labels/val/{label_name}"
        save_yolo_label(label_path, ann_data["polygons"], ann_data["width"], ann_data["height"])
    print(f"Dataset prepared: {len(train_items)} train, {len(val_items)} val images")

if __name__ == "__main__":
    prepare_yolo_dataset()
