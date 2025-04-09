import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random

image_folder = "datas/ds_imgs/"
annotations_path = "annotations_Electrode_GroovCenter.xml"
output_base_dir = "dataset_45"
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
        if abs(p1[0] - p2[0]) < 2:
            dx = 1
            adjusted.append((p1[0] + dx, p1[1]))
            adjusted.append((p2[0] - dx, p2[1]))
            count_adjusted += 1
    adjusted.append(points[-1])
    return adjusted, count_adjusted

def rotate_point(x, y, cx, cy, angle_deg):
    '''
    instead will rotate using opencv
    '''

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x -= cx
    y -= cy
    x_new = x * cos_a + y * sin_a
    y_new = -x * sin_a + y * cos_a
    return x_new + cx, y_new + cy

def transform_points_with_matrix(points, rot_matrix):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    rotated = cv2.transform(pts, rot_matrix)
    return [tuple(p[0]) for p in rotated]

def rotate_points(points, width, height, angle_deg):
    cx, cy = width / 2, height / 2
    return [rotate_point(x, y, cx, cy, angle_deg) for x, y in points]

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rot_matrix[0, 0])
    sin = abs(rot_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, rot_matrix, (new_w, new_h)), rot_matrix, new_w, new_h

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
            # cv2.imwrite(dst_img, cv2.imread(img_path))
            img = cv2.imread(img_path)
            # rotated_img, _, new_w, new_h = rotate_image(img, 45)
            rotated_img, rot_matrix, new_w, new_h = rotate_image(img, 45)
            cv2.imwrite(dst_img, rotated_img)

        for poly in image_tag.findall("polyline"):
            label = poly.attrib["label"]
            # pts = parse_points(poly.attrib["points"])
            # orig_pts = parse_points(poly.attrib["points"])
            # pts = rotate_points(orig_pts, w, h, 45)
            # pts, count = adjust_polyline_thickness(pts)
            orig_pts = parse_points(poly.attrib["points"])
            rotated_pts = transform_points_with_matrix(orig_pts, rot_matrix)
            rotated_pts, count = adjust_polyline_thickness(rotated_pts)
            adjusted_lines += count
            if label == "Electrode":
                # save_yolo_segmentation(dst_lbl, 0, pts, w, h)
                save_yolo_segmentation(dst_lbl, 0, rotated_pts, new_w, new_h)
            elif label == "groove center":
                # save_yolo_segmentation(dst_lbl, 1, pts, w, h)
                save_yolo_segmentation(dst_lbl, 1, rotated_pts, new_w, new_h)

process(train_entries, train_image_dir, train_label_dir)
process(val_entries, val_image_dir, val_label_dir)
print(f"Adjusted lines: {adjusted_lines}, Train: {len(train_entries)}, Val: {len(val_entries)}")