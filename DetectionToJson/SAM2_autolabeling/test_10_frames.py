#!/usr/bin/env python3
"""
TEST SCRIPT - Process only 10 frames and save annotations immediately
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import SAM

def parse_annotations():
    """Get the groove center points from your XML"""
    xml_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/annotations.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    points = []
    for image_tag in root.findall(".//image"):
        filename = image_tag.attrib['name']
        if "basler_recording_20250710_092901" in filename:  # Use this video's examples
            for poly_tag in image_tag.findall(".//polygon"):
                if poly_tag.attrib['label'] == 'groove center':
                    points_str = poly_tag.attrib['points']
                    polygon_points = []
                    for point_str in points_str.split(';'):
                        x, y = map(float, point_str.split(','))
                        polygon_points.append([x, y])
                    
                    # Get center point
                    center_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                    center_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                    points.append([center_x, center_y])
                    break
    
    return points

def mask_to_polygon(mask):
    """Convert mask to polygon like your XML annotations"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    points = []
    for point in simplified:
        x, y = point[0]
        points.append([float(x), float(y)])
    
    return points if len(points) >= 3 else None

def test_10_frames():
    """Test SAM2 on only 10 frames"""
    
    # Get test video
    video_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/basler_recording_20250710_092901.avi"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Get prompts from your annotations
    prompt_points = parse_annotations()
    if not prompt_points:
        print("No prompt points found!")
        return
    
    print(f"Using {len(prompt_points)} prompt points: {prompt_points}")
    
    # Initialize SAM2
    sam = SAM("sam2_b.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return
    
    # Process only 10 frames
    annotations = []
    
    for frame_idx in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_idx}...")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use first prompt point
        prompt_point = prompt_points[0]
        
        try:
            # Run SAM2 with point prompt
            results = sam(frame_rgb, points=[prompt_point], labels=[1])
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[0].cpu().numpy()
                    
                    # Convert to polygon like your XML
                    polygon = mask_to_polygon(mask)
                    
                    if polygon:
                        annotation = {
                            'frame': frame_idx,
                            'polygon_points': polygon,
                            'mask_area': int(np.sum(mask)),
                            'prompt_used': prompt_point
                        }
                        annotations.append(annotation)
                        print(f"  ✅ Frame {frame_idx}: Got polygon with {len(polygon)} points")
                    else:
                        print(f"  ❌ Frame {frame_idx}: No polygon extracted")
                else:
                    print(f"  ❌ Frame {frame_idx}: No masks in result")
            else:
                print(f"  ❌ Frame {frame_idx}: No results")
                
        except Exception as e:
            print(f"  ❌ Frame {frame_idx}: Error - {e}")
    
    cap.release()
    
    # Save results immediately
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_10_FRAMES_ANNOTATIONS.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'test_video': video_path,
            'total_frames_processed': 10,
            'successful_annotations': len(annotations),
            'prompt_points_used': prompt_points,
            'annotations': annotations
        }, f, indent=2)
    
    print(f"\n🎉 SAVED {len(annotations)} ANNOTATIONS TO: {output_file}")
    
    # Also save in XML format like yours
    save_xml_format(annotations, output_file.replace('.json', '.xml'))

def save_xml_format(annotations, xml_file):
    """Save in XML format similar to your annotations.xml"""
    root = ET.Element("annotations")
    
    for ann in annotations:
        image_elem = ET.SubElement(root, "image")
        image_elem.set("id", str(ann['frame']))
        image_elem.set("name", f"frame_{ann['frame']:04d}.png")
        image_elem.set("width", "1624")  # Your video width
        image_elem.set("height", "1234")  # Your video height
        
        polygon_elem = ET.SubElement(image_elem, "polygon")
        polygon_elem.set("label", "groove center")
        polygon_elem.set("source", "sam2_auto")
        polygon_elem.set("occluded", "0")
        
        # Convert points to string format like your XML
        points_str = ";".join([f"{p[0]:.2f},{p[1]:.2f}" for p in ann['polygon_points']])
        polygon_elem.set("points", points_str)
        polygon_elem.set("z_order", "0")
    
    # Write XML
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)
    print(f"🎉 ALSO SAVED XML FORMAT TO: {xml_file}")

if __name__ == "__main__":
    print("=== TESTING SAM2 ON 10 FRAMES ONLY ===")
    test_10_frames()
