import os
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image
from pathlib import Path

# CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT_PATH = "sam_vit_b.pth"  # download from Meta if not already
MODEL_TYPE = "vit_b"

FOLDER_IMAGES_TO_ANNOTATE = "sam_autolabel_example/images_to_annotate"
FOLDER_EXAMPLES = "sam_autolabel_example/example_annotated"
ANNOTATIONS_XML = "sam_autolabel_example/annotations/annotations.xml"

OUTPUT_LABELS_FOLDER = "sam_autolabel_example/output_labels_yolo"
os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)

# LOAD SAM MODEL
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# PARSE CVAT XML POLYGON MASKS
def parse_cvat_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    for image_tag in root.findall(".//image"):
        name = image_tag.attrib['name']
        polygons = []
        for poly in image_tag.findall(".//polygon"):
            points = poly.attrib['points']
            coords = [(float(x), float(y)) for x, y in (p.split(',') for p in points.split(';'))]
            polygons.append(coords)
        if polygons:
            annotations[name] = polygons
    return annotations

# Load example masks
example_masks = parse_cvat_annotations(ANNOTATIONS_XML)

# For each example image, get mask embedding
example_embeddings = []
for filename, polygons in example_masks.items():
    path = os.path.join(FOLDER_EXAMPLES, filename)
    if not os.path.exists(path):
        continue
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    for poly in polygons:
        input_point = np.array(poly).mean(axis=0)[None, :]
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        embedding = predictor.get_image_embedding().cpu()
        example_embeddings.append((embedding, masks[0]))

# Annotate unlabelled images
image_paths = glob.glob(os.path.join(FOLDER_IMAGES_TO_ANNOTATE, "*.png"))
for img_path in image_paths:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    predictor.set_image(img_rgb)

    input_point = np.array([[width // 2, height // 2]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    # Find contours for YOLO polygon annotation
    contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(OUTPUT_LABELS_FOLDER, label_file)

    with open(label_path, 'w') as f:
        for contour in contours:
            points = contour.squeeze().astype(np.float32)
            if points.ndim != 2 or len(points) < 3:
                continue
            norm_points = [(x / width, y / height) for x, y in points]
            flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_points)
            f.write(f"0 {flat}\n")

    print(f"Saved YOLO label: {label_path}")
