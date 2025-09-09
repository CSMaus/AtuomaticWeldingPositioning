import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import random

def parse_xml_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    
    for image in root.findall('.//image'):
        image_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        
        polygons = []
        for polygon in image.findall('polygon'):
            points_str = polygon.get('points')
            label = polygon.get('label', 'weld')
            points = [tuple(map(float, p.split(','))) for p in points_str.strip().split(';')]
            polygons.append({'points': points, 'label': label})
        
        annotations[image_name] = {
            'width': width,
            'height': height,
            'polygons': polygons
        }
    
    return annotations

def normalize_points(points, width, height):
    return [(x / width, y / height) for x, y in points]

def save_yolo_label(label_path, polygons, width, height):
    with open(label_path, 'w') as f:
        for polygon in polygons:
            points = polygon['points']
            norm_points = normalize_points(points, width, height)
            flat_points = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
            line = f"0 " + " ".join(flat_points) + "\n"
            f.write(line)

def prepare_yolo_dataset():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(script_dir, "dataset")
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    # Process each annotation file
    annotation_files = [
        ("annotations-basler_recordings_frames.xml", "basler_recordings_frames"),
        ("annotations-br_frames-0909.xml", "br_frames-0909")
    ]
    
    all_images = []
    
    for ann_file, img_folder in annotation_files:
        ann_path = os.path.join(data_dir, ann_file)
        img_dir = os.path.join(data_dir, img_folder)
        
        if not os.path.exists(ann_path) or not os.path.exists(img_dir):
            print(f"Skipping {ann_file} - path not found")
            continue
            
        annotations = parse_xml_annotations(ann_path)
        
        for img_name, ann_data in annotations.items():
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                all_images.append((img_path, ann_data, img_name))
    
    # Split dataset
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Process training set
    print("Processing training set...")
    for img_path, ann_data, img_name in tqdm(train_images):
        # Copy image
        dst_img = f"{output_dir}/images/train/{img_name}"
        shutil.copy2(img_path, dst_img)
        
        # Save label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/train/{label_name}"
        save_yolo_label(label_path, ann_data['polygons'], ann_data['width'], ann_data['height'])
    
    # Process validation set
    print("Processing validation set...")
    for img_path, ann_data, img_name in tqdm(val_images):
        # Copy image
        dst_img = f"{output_dir}/images/val/{img_name}"
        shutil.copy2(img_path, dst_img)
        
        # Save label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/val/{label_name}"
        save_yolo_label(label_path, ann_data['polygons'], ann_data['width'], ann_data['height'])
    
    print(f"Dataset prepared: {len(train_images)} train, {len(val_images)} val images")

if __name__ == "__main__":
    prepare_yolo_dataset()
