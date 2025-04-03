# from CVAT xml file extract masks from polylines as contours
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd


image_folder = "ds_collect/"
annotations_path = "annotations_Electrode_GroovCenter.xml"
mask_folder = "masks/"
os.makedirs(mask_folder, exist_ok=True)

def parse_points(points_str):
    return [tuple(map(float, p.split(','))) for p in points_str.strip().split(';')]

def draw_polyline_mask(img_shape, points, thickness):
    mask = np.zeros(img_shape, dtype=np.uint8)
    int_points = np.array(points, dtype=np.int32)
    cv2.polylines(mask, [int_points], isClosed=False, color=255, thickness=thickness)
    return mask

def extract_masks_from_xml(xml_path, output_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image_tag in root.findall("image"):
        filename = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))
        electrode_mask = np.zeros((height, width), dtype=np.uint8)
        groove_mask = np.zeros((height, width), dtype=np.uint8)

        for polyline in image_tag.findall("polyline"):
            label = polyline.attrib["label"]
            points = parse_points(polyline.attrib["points"])

            if label == "Electrode":
                mask = draw_polyline_mask((height, width), points, thickness=3)
                electrode_mask = cv2.bitwise_or(electrode_mask, mask)
            elif label == "groove center":
                mask = draw_polyline_mask((height, width), points, thickness=4)
                groove_mask = cv2.bitwise_or(groove_mask, mask)

        if np.any(electrode_mask):
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_electrode.png"), electrode_mask)
        if np.any(groove_mask):
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_groove_center.png"), groove_mask)


extract_masks_from_xml(annotations_path, mask_folder)
'''def draw_polyline_on_mask(mask, points_str, thickness):
    points = [tuple(map(float, pt.split(","))) for pt in points_str.strip().split(";")]
    for i in range(len(points) - 1):
        pt1 = tuple(map(int, points[i]))
        pt2 = tuple(map(int, points[i + 1]))
        cv2.line(mask, pt1, pt2, color=255, thickness=thickness)
    return mask

tree = ET.parse(xml_path)
root = tree.getroot()

for image_tag in tqdm(root.findall("image")):
    filename = image_tag.attrib["name"]
    width = int(image_tag.attrib["width"])
    height = int(image_tag.attrib["height"])
    mask = np.zeros((height, width), dtype=np.uint8)

    for polyline in image_tag.findall("polyline"):
        label = polyline.attrib["label"]
        points = polyline.attrib["points"]
        if label == "groove center":
            mask = draw_polyline_on_mask(mask, points, thickness=4)
        elif label == "Electrode":
            mask = draw_polyline_on_mask(mask, points, thickness=3)

    mask_path = os.path.join(output_mask_folder, os.path.splitext(filename)[0] + "_mask.png")
    cv2.imwrite(mask_path, mask)

import ace_tools as tools; tools.display_dataframe_to_user(name="Example Mask Files", dataframe=
    pd.DataFrame({"Generated Masks": os.listdir(output_mask_folder)}).head(10)
)'''


