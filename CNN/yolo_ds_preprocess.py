import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

image_folder = "datas/ds_imgs/"
annotations_path = "annotations_Electrode_GroovCenter.xml"
output_image_dir = "dataset/images/train/"
output_label_dir = "dataset/labels/train/"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

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

tree = ET.parse(annotations_path)
root = tree.getroot()

for image_tag in tqdm(root.findall("image")):
    filename = image_tag.get("name")
    width = int(image_tag.get("width"))
    height = int(image_tag.get("height"))
    src_img_path = os.path.join(image_folder, filename)
    dst_img_path = os.path.join(output_image_dir, filename)
    label_path = os.path.join(output_label_dir, os.path.splitext(filename)[0] + ".txt")

    if os.path.exists(src_img_path):
        img = cv2.imread(src_img_path)
        cv2.imwrite(dst_img_path, img)

    for polyline in image_tag.findall("polyline"):
        label = polyline.attrib["label"]
        points = parse_points(polyline.attrib["points"])
        if label == "Electrode":
            save_yolo_segmentation(label_path, 0, points, width, height)
        elif label == "groove center":
            save_yolo_segmentation(label_path, 1, points, width, height)
