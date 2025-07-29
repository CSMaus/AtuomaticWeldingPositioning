#!/usr/bin/env python3
"""
Use your EXACT polygons as prompts to get IDENTICAL results
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import SAM

def get_your_exact_polygons():
    """Get YOUR exact polygons from XML"""
    xml_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/annotations.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    polygons = {}
    for image_tag in root.findall(".//image"):
        filename = image_tag.attrib['name']
        if "basler_recording_20250710_092901" in filename:
            for poly_tag in image_tag.findall(".//polygon"):
                if poly_tag.attrib['label'] == 'groove center':
                    points_str = poly_tag.attrib['points']
                    polygon_points = []
                    for point_str in points_str.split(';'):
                        x, y = map(float, point_str.split(','))
                        polygon_points.append([x, y])
                    
                    polygons[filename] = polygon_points
                    break
    
    return polygons

def polygon_to_mask(polygon, width=1624, height=1234):
    """Convert your polygon to a binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def get_multiple_points_from_polygon(polygon, num_points=5):
    """Get multiple points from your polygon boundary"""
    points = []
    polygon_array = np.array(polygon)
    
    # Get points along the polygon boundary
    for i in range(0, len(polygon), max(1, len(polygon) // num_points)):
        points.append(polygon[i])
    
    # Add center point
    center_x = np.mean(polygon_array[:, 0])
    center_y = np.mean(polygon_array[:, 1])
    points.append([center_x, center_y])
    
    return points

def test_with_your_polygons():
    """Test using YOUR polygons as reference"""
    
    # Get your exact polygons
    your_polygons = get_your_exact_polygons()
    if not your_polygons:
        print("No polygons found!")
        return
    
    print(f"Found {len(your_polygons)} of your polygons")
    
    # Get first polygon as reference
    first_file = list(your_polygons.keys())[0]
    reference_polygon = your_polygons[first_file]
    
    print(f"Using reference polygon from: {first_file}")
    print(f"Polygon has {len(reference_polygon)} points")
    
    # Get multiple prompt points from your polygon
    prompt_points = get_multiple_points_from_polygon(reference_polygon, num_points=8)
    
    print(f"Using {len(prompt_points)} prompt points from your polygon")
    
    # Initialize SAM2
    sam = SAM("sam2_b.pt")
    
    # Test video
    video_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/basler_recording_20250710_092901.avi"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return
    
    annotations = []
    
    # Process 10 frames
    for frame_idx in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_idx}...")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Use multiple points from your polygon as prompts
            results = sam(frame_rgb, 
                         points=prompt_points, 
                         labels=[1] * len(prompt_points))  # All positive
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[0].cpu().numpy()
                    
                    # Convert to polygon
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        polygon = []
                        for point in simplified:
                            x, y = point[0]
                            polygon.append([float(x), float(y)])
                        
                        if len(polygon) >= 3:
                            annotation = {
                                'frame': frame_idx,
                                'polygon_points': polygon,
                                'your_reference_polygon': reference_polygon,
                                'prompt_points_used': prompt_points,
                                'mask_area': int(np.sum(mask))
                            }
                            annotations.append(annotation)
                            print(f"  ✅ Frame {frame_idx}: Got polygon with {len(polygon)} points")
                        else:
                            print(f"  ❌ Frame {frame_idx}: Polygon too small")
                    else:
                        print(f"  ❌ Frame {frame_idx}: No contours")
                else:
                    print(f"  ❌ Frame {frame_idx}: No masks")
            else:
                print(f"  ❌ Frame {frame_idx}: No results")
                
        except Exception as e:
            print(f"  ❌ Frame {frame_idx}: Error - {e}")
    
    cap.release()
    
    # Save results
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_POLYGON_PROMPTS.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'method': 'multiple_points_from_your_polygon',
            'reference_file': first_file,
            'reference_polygon': reference_polygon,
            'prompt_points': prompt_points,
            'total_frames_processed': 10,
            'successful_annotations': len(annotations),
            'annotations': annotations
        }, f, indent=2)
    
    print(f"\n🎉 SAVED {len(annotations)} ANNOTATIONS TO: {output_file}")

if __name__ == "__main__":
    print("=== TESTING WITH YOUR EXACT POLYGONS AS PROMPTS ===")
    test_with_your_polygons()
