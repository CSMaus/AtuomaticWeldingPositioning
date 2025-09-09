#!/usr/bin/env python3
"""
Use ALL annotated frames as prompts for SAM2
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import SAM

TOTAL_FRAMES_TO_PROCESS = 10

def get_all_annotations_for_video(video_name):
    """Get ALL annotations for the specific video"""
    xml_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/annotations.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    for image_tag in root.findall(".//image"):
        filename = image_tag.attrib['name']
        if video_name in filename:
            for poly_tag in image_tag.findall(".//polygon"):
                if poly_tag.attrib['label'] == 'groove center':
                    points_str = poly_tag.attrib['points']
                    polygon_points = []
                    for point_str in points_str.split(';'):
                        x, y = map(float, point_str.split(','))
                        polygon_points.append([x, y])
                    
                    annotations.append({
                        'filename': filename,
                        'polygon': polygon_points
                    })
                    break
    
    return annotations

def get_all_prompt_points(all_annotations, num_points_per_polygon=8):
    """Get comprehensive prompt points from ALL annotations"""
    positive_points = []
    negative_points = []
    
    for ann in all_annotations:
        polygon = ann['polygon']
        polygon_array = np.array(polygon)
        
        # Get MORE points along the polygon boundary
        for i in range(0, len(polygon), max(1, len(polygon) // num_points_per_polygon)):
            positive_points.append(polygon[i])
        
        # Add center point
        center_x = np.mean(polygon_array[:, 0])
        center_y = np.mean(polygon_array[:, 1])
        positive_points.append([center_x, center_y])
        
        # Add interior points to cover the FULL area
        min_x, max_x = np.min(polygon_array[:, 0]), np.max(polygon_array[:, 0])
        min_y, max_y = np.min(polygon_array[:, 1]), np.max(polygon_array[:, 1])
        
        # Add grid of interior points
        for i in range(3):
            for j in range(3):
                interior_x = min_x + (max_x - min_x) * (i + 1) / 4
                interior_y = min_y + (max_y - min_y) * (j + 1) / 4
                
                # Check if point is inside polygon
                if cv2.pointPolygonTest(polygon_array.astype(np.int32), (interior_x, interior_y), False) >= 0:
                    positive_points.append([interior_x, interior_y])
        
        # Add negative points OUTSIDE the polygon
        # Expand bounding box and add points outside
        expand = 50  # pixels
        outside_points = [
            [min_x - expand, min_y - expand],  # Top-left outside
            [max_x + expand, min_y - expand],  # Top-right outside
            [min_x - expand, max_y + expand],  # Bottom-left outside
            [max_x + expand, max_y + expand],  # Bottom-right outside
        ]
        
        for point in outside_points:
            if cv2.pointPolygonTest(polygon_array.astype(np.int32), tuple(point), False) < 0:
                negative_points.append(point)
    
    return positive_points, negative_points

def load_existing_annotations(output_file):
    """Load existing annotations if file exists"""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        print(f"Found existing annotations: {len(data.get('annotations', []))}")
        return data
    return None

def save_annotations_incrementally(data, output_file):
    """Save annotations to file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def process_video():
    """Process video using ALL annotations as prompts"""
    
    video_name = "basler_recording_20250710_092901"
    video_path = f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/{video_name}.avi"
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_POLYGON_PROMPTS.json"
    
    # Get ALL annotations for this video
    all_annotations = get_all_annotations_for_video(video_name)
    if not all_annotations:
        print("No annotations found!")
        return
    
    print(f"Found {len(all_annotations)} annotated frames for {video_name}")
    for ann in all_annotations:
        print(f"  - {ann['filename']}")
    
    # Get ALL prompt points from ALL annotations
    positive_points, negative_points = get_all_prompt_points(all_annotations, num_points_per_polygon=8)
    all_prompt_points = positive_points + negative_points
    all_labels = [1] * len(positive_points) + [0] * len(negative_points)
    
    print(f"Using {len(positive_points)} positive + {len(negative_points)} negative = {len(all_prompt_points)} total prompt points")
    
    # Load existing annotations
    existing_data = load_existing_annotations(output_file)
    if existing_data:
        annotations = existing_data.get('annotations', [])
        existing_frames = {ann['frame'] for ann in annotations}
    else:
        annotations = []
        existing_frames = set()
    
    # Determine which frames to process
    frames_to_process = [f for f in range(TOTAL_FRAMES_TO_PROCESS) if f not in existing_frames]
    
    if not frames_to_process:
        print(f"All {TOTAL_FRAMES_TO_PROCESS} frames already processed!")
        return
    
    print(f"Processing {len(frames_to_process)} new frames...")
    
    # Initialize SAM2
    sam = SAM("sam2_b.pt")
    
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
            # Load first example image and its mask
            first_annotation = all_annotations[0]
            example_image_path = f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings_frames/{first_annotation['filename']}"
            
            if os.path.exists(example_image_path):
                example_image = cv2.imread(example_image_path)
                example_image_rgb = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
                
                # Create mask from polygon
                height, width = example_image.shape[:2]
                example_mask = np.zeros((height, width), dtype=np.uint8)
                points = np.array(first_annotation['polygon'], dtype=np.int32)
                cv2.fillPoly(example_mask, [points], 255)
                
                # Use example image with mask
                results = sam(frame_rgb, masks=[example_mask])
            else:
                # Fallback to points if image not found
                results = sam(frame_rgb, 
                             points=all_prompt_points, 
                             labels=all_labels)
            
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
                                'prompt_points_used': all_prompt_points,
                                'prompt_labels_used': all_labels,
                                'positive_prompts': len(positive_points),
                                'negative_prompts': len(negative_points),
                                'total_reference_frames': len(all_annotations),
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
                'method': 'all_annotations_as_prompts',
                'video_name': video_name,
                'video_path': video_path,
                'total_reference_frames': len(all_annotations),
                'total_prompt_points': len(all_prompt_points),
                'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
                'successful_annotations': len(annotations),
                'annotations': annotations
            }
            save_annotations_incrementally(data, output_file)
            print(f"  💾 Saved {processed_count} annotations")
    
    cap.release()
    
    # Final save
    data = {
        'method': 'all_annotations_as_prompts',
        'video_name': video_name,
        'video_path': video_path,
        'total_reference_frames': len(all_annotations),
        'total_prompt_points': len(all_prompt_points),
        'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
        'successful_annotations': len(annotations),
        'annotations': annotations
    }
    save_annotations_incrementally(data, output_file)
    
    print(f"\n🎉 COMPLETE!")
    print(f"Used {len(all_annotations)} reference frames")
    print(f"Used {len(all_prompt_points)} total prompt points")
    print(f"Processed {processed_count} new frames")

if __name__ == "__main__":
    print("=== SAM2 WITH ALL ANNOTATIONS AS PROMPTS ===")
    process_video()
