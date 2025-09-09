#!/usr/bin/env python3
"""
Use ALL your annotated frames as reference examples for SAM2
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import SAM

# CONFIGURATION
TOTAL_FRAMES_TO_PROCESS = 400

def get_all_your_annotations():
    """Get ALL your groove center annotations from XML"""
    xml_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/annotations.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
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
                    
                    # Extract frame number from filename
                    frame_num = int(filename.split('-frame_')[1].split('.')[0])
                    
                    annotations[frame_num] = {
                        'filename': filename,
                        'polygon': polygon_points
                    }
                    break
    
    return annotations

def get_multiple_prompts_from_all_annotations(all_annotations):
    """Get prompt points from ALL your annotations"""
    all_prompt_points = []
    
    for frame_num, ann_data in all_annotations.items():
        polygon = ann_data['polygon']
        polygon_array = np.array(polygon)
        
        # Get center point
        center_x = np.mean(polygon_array[:, 0])
        center_y = np.mean(polygon_array[:, 1])
        all_prompt_points.append([center_x, center_y])
        
        # Get a few boundary points
        for i in range(0, len(polygon), max(1, len(polygon) // 3)):
            all_prompt_points.append(polygon[i])
    
    return all_prompt_points

def load_existing_annotations(output_file):
    """Load existing annotations if file exists"""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        print(f"Found existing annotations: {len(data.get('annotations', []))}")
        return data
    return None

def test_with_multiple_references():
    """Use ALL your annotations as reference examples"""
    
    # Get ALL your annotations
    all_your_annotations = get_all_your_annotations()
    if not all_your_annotations:
        print("No annotations found!")
        return
    
    print(f"Found {len(all_your_annotations)} reference annotations at frames: {sorted(all_your_annotations.keys())}")
    
    # Get prompt points from ALL annotations
    all_prompt_points = get_multiple_prompts_from_all_annotations(all_your_annotations)
    print(f"Using {len(all_prompt_points)} prompt points from ALL your annotations")
    
    # Video setup
    video_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/basler_recording_20250710_092901.avi"
    video_name = os.path.basename(video_path).replace('.avi', '')
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_MULTIPLE_REFERENCES.json"
    
    # Load existing
    existing_data = load_existing_annotations(output_file)
    if existing_data:
        annotations = existing_data.get('annotations', [])
        existing_frames = {ann['frame'] for ann in annotations}
    else:
        annotations = []
        existing_frames = set()
    
    # Frames to process
    frames_to_process = [f for f in range(TOTAL_FRAMES_TO_PROCESS) if f not in existing_frames]
    
    if not frames_to_process:
        print(f"All {TOTAL_FRAMES_TO_PROCESS} frames already processed!")
        return
    
    print(f"Processing {len(frames_to_process)} new frames...")
    
    # Initialize SAM2
    sam = SAM("sam2.1_b.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return
    
    processed_count = 0
    
    for frame_idx in frames_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        print(f"Processing frame {frame_idx}... ({processed_count + 1}/{len(frames_to_process)})")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Use ALL prompt points from ALL your annotations
            results = sam(frame_rgb, 
                         points=all_prompt_points, 
                         labels=[1] * len(all_prompt_points))
            
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
                                'reference_frames': list(all_your_annotations.keys()),
                                'total_prompt_points': len(all_prompt_points),
                                'mask_area': int(np.sum(mask))
                            }
                            annotations.append(annotation)
                            processed_count += 1
                            
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
        
        # Save every 20 frames
        if processed_count % 20 == 0 and processed_count > 0:
            data = {
                'method': 'multiple_reference_annotations',
                'video_name': video_name,
                'video_path': video_path,
                'reference_annotations': all_your_annotations,
                'total_prompt_points': len(all_prompt_points),
                'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
                'successful_annotations': len(annotations),
                'annotations': annotations
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  💾 Saved {processed_count} annotations")
    
    cap.release()
    
    # Final save
    data = {
        'method': 'multiple_reference_annotations',
        'video_name': video_name,
        'video_path': video_path,
        'reference_annotations': all_your_annotations,
        'total_prompt_points': len(all_prompt_points),
        'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
        'successful_annotations': len(annotations),
        'annotations': annotations
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n🎉 COMPLETE! Processed {processed_count} new frames")
    print(f"Used {len(all_your_annotations)} reference annotations")
    print(f"Total prompt points: {len(all_prompt_points)}")

if __name__ == "__main__":
    print("=== USING ALL YOUR ANNOTATIONS AS REFERENCES ===")
    test_with_multiple_references()
