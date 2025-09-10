import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import random



CLASS_MAP = {
    "groove center": 0,
    "Electrode": 1
}


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
            label = polygon['label']
            class_id = CLASS_MAP.get(label, None)
            if class_id is None:
                continue  # skip unknown labels

            norm_points = normalize_points(polygon['points'], width, height)
            flat_points = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
            line = f"{class_id} " + " ".join(flat_points) + "\n"
            f.write(line)


def prepare_yolo_dataset():
    #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(script_dir, "dataset")
    
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    annotation_files = [
        ("annotations-basler_recordings_frames-el.xml", "basler_recordings_frames"),
        ("annotations-br_frames-0909-el.xml", "br_frames-0909"),
        ("annotations-labeling data-0910-el.xml", "labeling data-0910")
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
    
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print("Processing training set...")
    for img_path, ann_data, img_name in tqdm(train_images):
        dst_img = f"{output_dir}/images/train/{img_name}"
        shutil.copy2(img_path, dst_img)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/train/{label_name}"
        save_yolo_label(label_path, ann_data['polygons'], ann_data['width'], ann_data['height'])
    
    print("Processing validation set...")
    for img_path, ann_data, img_name in tqdm(val_images):
        dst_img = f"{output_dir}/images/val/{img_name}"
        shutil.copy2(img_path, dst_img)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/val/{label_name}"
        save_yolo_label(label_path, ann_data['polygons'], ann_data['width'], ann_data['height'])
    
    print(f"Dataset prepared: {len(train_images)} train, {len(val_images)} val images")

if __name__ == "__main__":
    prepare_yolo_dataset()
