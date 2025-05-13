import os
import sys

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
from pathlib import Path


path = os.path.join(Path.cwd().parents[3], "data/")
folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

image_folder = os.path.join(path, "RL_Groove_Rod-Data/labeling data")
print(image_folder)

annotations_path = "labeling data - all.xml"
output_base_dir = "dataset_labeling"
train_image_dir = os.path.join(output_base_dir, "images/train/")
val_image_dir = os.path.join(output_base_dir, "images/val/")
train_label_dir = os.path.join(output_base_dir, "labels/train/")
val_label_dir = os.path.join(output_base_dir, "labels/val/")
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

def parse_points(points_str):
    return [tuple(map(float, p.split(','))) for p in points_str.strip().split(';')]

def normalize_points(points, width, height):
    return [(x / width, y / height) for x, y in points]

def save_yolo_segmentation(label_file, class_id, points, width, height):
    norm_points = normalize_points(points, width, height)
    flat = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
    line = f"{class_id} " + " ".join(flat) + "\n"
    with open(label_file, "a") as f:
        f.write(line)

def adjust_polyline_thickness(points):
    adjusted = []
    count_adjusted = 0
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        adjusted.append(p1)
        if abs(p1[0] - p2[0]) < 3:
            dx = 1
            adjusted.append((p1[0] + dx, p1[1]))
            adjusted.append((p2[0] - dx, p2[1]))
            count_adjusted += 1
    adjusted.append(points[-1])
    return adjusted, count_adjusted

tree = ET.parse(annotations_path)
root = tree.getroot()

image_entries = root.findall("image")
random.shuffle(image_entries)
split_index = int(0.85 * len(image_entries))
train_entries = image_entries[:split_index]
val_entries = image_entries[split_index:]

adjusted_lines = 0

def process(entries, image_dir, label_dir):
    global adjusted_lines
    for image_tag in tqdm(entries):
        fname = image_tag.get("name")
        w, h = int(image_tag.get("width")), int(image_tag.get("height"))
        img_path = os.path.join(image_folder, fname)
        dst_img = os.path.join(image_dir, fname)
        dst_lbl = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        if os.path.exists(img_path):
            cv2.imwrite(dst_img, cv2.imread(img_path))
        for poly in image_tag.findall("polygon"):
            label = poly.attrib["label"]
            pts = parse_points(poly.attrib["points"])
            if label == "W-Rod":
                save_yolo_segmentation(dst_lbl, 0, pts, w, h)
            elif label in ["L-Groove", "R-Groove"]:
                save_yolo_segmentation(dst_lbl, 1, pts, w, h)

process(train_entries, train_image_dir, train_label_dir)
process(val_entries, val_image_dir, val_label_dir)
print(f"Adjusted lines: {adjusted_lines}, Train: {len(train_entries)}, Val: {len(val_entries)}")
