#!/usr/bin/env python3
"""
FIXED: Use closest reference annotation for each frame (but never the same frame)
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import SAM

TOTAL_FRAMES_TO_PROCESS = 10

def get_your_exact_polygons_with_frames():
    """Get YOUR exact polygons from XML with frame numbers"""
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
                    
                    # Extract frame number from filename
                    frame_num = int(filename.split('-frame_')[1].split('.')[0])
                    polygons[frame_num] = {
                        'filename': filename,
                        'polygon': polygon_points
                    }
                    break
    
    return polygons

def get_closest_reference_for_frame(frame_idx, all_polygons):
    """Get the closest reference annotation (but NOT the same frame)"""
    if not all_polygons:
        return None, None
    
    annotated_frames = list(all_polygons.keys())
    
    # Remove the current frame from candidates (can't use it to annotate itself!)
    available_frames = [f for f in annotated_frames if f != frame_idx]
    
    if not available_frames:
        # If no other frames available, use the first one
        closest_frame = annotated_frames[0]
    else:
        # Find closest frame
        closest_frame = min(available_frames, key=lambda x: abs(x - frame_idx))
    
    return all_polygons[closest_frame]['polygon'], closest_frame

def load_existing_annotations(output_file):
    """Load existing annotations if file exists"""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        print(f"Found existing annotations file with {len(data.get('annotations', []))} annotations")
        return data
    else:
        print("No existing annotations file found, starting fresh")
        return None

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

def save_annotations_incrementally(data, output_file):
    """Save annotations to file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def test_with_closest_polygons():
    """Test using closest reference polygon for each frame"""
    
    your_polygons = get_your_exact_polygons_with_frames()
    if not your_polygons:
        print("No polygons found!")
        return
    
    annotated_frames = sorted(your_polygons.keys())
    print(f"Found {len(your_polygons)} reference annotations at frames: {annotated_frames}")
    
    video_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/basler_recording_20250710_092901.avi"
    video_name = os.path.basename(video_path).replace('.avi', '')
    
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_POLYGON_PROMPTS.json"
    
    existing_data = load_existing_annotations(output_file)
    
    if existing_data:
        annotations = existing_data.get('annotations', [])
        existing_frames = {ann['frame'] for ann in annotations}
        print(f"Already have annotations for frames: {sorted(existing_frames)}")
    else:
        annotations = []
        existing_frames = set()
    
    data = {
        'method': 'closest_reference_polygon',
        'video_name': video_name,
        'video_path': video_path,
        'reference_frames': annotated_frames,
        'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
        'total_frames_processed': len(existing_frames),
        'successful_annotations': len(annotations),
        'annotations': annotations
    }
    
    # Determine which frames to process
    frames_to_process = []
    for frame_idx in range(TOTAL_FRAMES_TO_PROCESS):
        if frame_idx not in existing_frames:
            frames_to_process.append(frame_idx)
    
    if not frames_to_process:
        print(f"✅ All {TOTAL_FRAMES_TO_PROCESS} frames already processed!")
        return
    
    print(f"Need to process {len(frames_to_process)} more frames")
    
    # Initialize SAM2
    sam = SAM("sam2_b.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return
    
    # Process only the missing frames
    processed_count = 0
    
    for frame_idx in frames_to_process:
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Cannot read frame {frame_idx}, stopping")
            break
        
        # Get the closest reference polygon for THIS frame
        reference_polygon, reference_frame = get_closest_reference_for_frame(frame_idx, your_polygons)
        if reference_polygon is None:
            print(f"  ❌ Frame {frame_idx}: No reference polygon")
            continue
        
        # Get prompt points from the closest reference
        prompt_points = get_multiple_points_from_polygon(reference_polygon, num_points=8)
        
        print(f"Processing frame {frame_idx}... using reference frame {reference_frame} ({processed_count + 1}/{len(frames_to_process)})")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Use points from the closest reference polygon
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
                                'reference_frame': reference_frame,
                                'reference_polygon': reference_polygon,
                                'prompt_points_used': prompt_points,
                                'mask_area': int(np.sum(mask))
                            }
                            annotations.append(annotation)
                            processed_count += 1
                            
                            # Update data
                            data['annotations'] = annotations
                            data['total_frames_processed'] = len(existing_frames) + processed_count
                            data['successful_annotations'] = len(annotations)
                            
                            # Save incrementally every 10 frames
                            if processed_count % 10 == 0:
                                save_annotations_incrementally(data, output_file)
                                print(f"  💾 Saved progress: {processed_count} new annotations")
                            
                            print(f"  ✅ Frame {frame_idx}: Got polygon with {len(polygon)} points (ref: {reference_frame})")
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
        
        # Save every 50 frames or if interrupted
        if processed_count % 50 == 0 and processed_count > 0:
            save_annotations_incrementally(data, output_file)
            print(f"  💾 Checkpoint: Saved {processed_count} new annotations")
    
    cap.release()
    
    # Final save
    data['annotations'] = annotations
    data['total_frames_processed'] = len(existing_frames) + processed_count
    data['successful_annotations'] = len(annotations)
    
    save_annotations_incrementally(data, output_file)
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"   - New annotations: {processed_count}")
    print(f"   - Total annotations: {len(annotations)}")
    print(f"   - Reference frames: {annotated_frames}")
    print(f"   - Saved to: {output_file}")
    print(f"   - Video: {video_name}")

if __name__ == "__main__":
    print("=== USING CLOSEST REFERENCE ANNOTATIONS (FIXED) ===")
    print(f"Configured to process {TOTAL_FRAMES_TO_PROCESS} total frames")
    print("Uses closest reference frame for each processed frame")
    test_with_closest_polygons()
