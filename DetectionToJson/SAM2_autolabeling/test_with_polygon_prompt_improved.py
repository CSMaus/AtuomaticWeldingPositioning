#!/usr/bin/env python3
"""
IMPROVED: Use closest reference annotations with interpolation
Central part works perfectly, now fixing outer contour prediction
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import SAM

# CONFIGURATION
TOTAL_FRAMES_TO_PROCESS = 10

def get_your_exact_polygons():
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

def get_multiple_points_from_polygon(polygon, num_points=8):
    """Get multiple points from polygon boundary"""
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

def get_best_reference_prompts_for_frame(frame_idx, all_polygons):
    """Get prompts from ONLY the 1-2 closest reference annotations"""
    if not all_polygons:
        return []
    
    annotated_frames = sorted(all_polygons.keys())
    
    # Find the 1 or 2 closest frames
    if frame_idx <= annotated_frames[0]:
        # Use ONLY first annotation
        closest_frame = annotated_frames[0]
        polygon = all_polygons[closest_frame]['polygon']
        print(f"    Using single reference: frame {closest_frame}")
        return get_multiple_points_from_polygon(polygon, num_points=8)
    
    elif frame_idx >= annotated_frames[-1]:
        # Use ONLY last annotation
        closest_frame = annotated_frames[-1]
        polygon = all_polygons[closest_frame]['polygon']
        print(f"    Using single reference: frame {closest_frame}")
        return get_multiple_points_from_polygon(polygon, num_points=8)
    
    else:
        # Find the 2 closest frames (before and after)
        before_frame = None
        after_frame = None
        
        for ann_frame in annotated_frames:
            if ann_frame <= frame_idx:
                before_frame = ann_frame
            elif ann_frame > frame_idx and after_frame is None:
                after_frame = ann_frame
                break
        
        # Calculate distances
        dist_before = abs(frame_idx - before_frame) if before_frame is not None else float('inf')
        dist_after = abs(frame_idx - after_frame) if after_frame is not None else float('inf')
        
        # If one frame is much closer, use only that one
        if dist_before < dist_after * 0.5:  # Before frame is much closer
            polygon = all_polygons[before_frame]['polygon']
            print(f"    Using single reference: frame {before_frame} (much closer)")
            return get_multiple_points_from_polygon(polygon, num_points=8)
        elif dist_after < dist_before * 0.5:  # After frame is much closer
            polygon = all_polygons[after_frame]['polygon']
            print(f"    Using single reference: frame {after_frame} (much closer)")
            return get_multiple_points_from_polygon(polygon, num_points=8)
        else:
            # Use BOTH closest frames (only 2!)
            before_polygon = all_polygons[before_frame]['polygon']
            after_polygon = all_polygons[after_frame]['polygon']
            
            before_points = get_multiple_points_from_polygon(before_polygon, num_points=4)
            after_points = get_multiple_points_from_polygon(after_polygon, num_points=4)
            
            print(f"    Using TWO references: frames {before_frame} and {after_frame}")
            return before_points + after_points

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

def test_with_improved_polygons():
    """Test using improved polygon reference selection"""
    
    # Get your exact polygons with frame numbers
    your_polygons = get_your_exact_polygons()
    if not your_polygons:
        print("No polygons found!")
        return
    
    annotated_frames = sorted(your_polygons.keys())
    print(f"Found {len(your_polygons)} reference annotations at frames: {annotated_frames}")
    
    # Video setup
    video_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/basler_recording_20250710_092901.avi"
    video_name = os.path.basename(video_path).replace('.avi', '')
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_POLYGON_PROMPTS.json"
    
    # Load existing annotations
    existing_data = load_existing_annotations(output_file)
    
    if existing_data:
        annotations = existing_data.get('annotations', [])
        existing_frames = {ann['frame'] for ann in annotations}
        print(f"Already have annotations for frames: {sorted(existing_frames)}")
    else:
        annotations = []
        existing_frames = set()
    
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
        
        # Get the best prompt points for this specific frame
        prompt_points = get_best_reference_prompts_for_frame(frame_idx, your_polygons)
        
        if not prompt_points:
            print(f"  ❌ Frame {frame_idx}: No prompt points")
            continue
        
        print(f"Processing frame {frame_idx}... ({processed_count + 1}/{len(frames_to_process)}) - {len(prompt_points)} prompts")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Use frame-specific prompt points
            results = sam(frame_rgb, 
                         points=prompt_points, 
                         labels=[1] * len(prompt_points))  # All positive
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[0].cpu().numpy()
                    
                    # Convert to polygon with higher precision
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.001 * cv2.arcLength(largest_contour, True)  # More precise
                        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        polygon = []
                        for point in simplified:
                            x, y = point[0]
                            polygon.append([float(x), float(y)])
                        
                        if len(polygon) >= 3:
                            annotation = {
                                'frame': frame_idx,
                                'polygon_points': polygon,
                                'prompt_points_used': prompt_points,
                                'reference_frames': annotated_frames,
                                'mask_area': int(np.sum(mask))
                            }
                            annotations.append(annotation)
                            processed_count += 1
                            
                            # Update data
                            data = {
                                'method': 'adaptive_polygon_prompts',
                                'video_name': video_name,
                                'video_path': video_path,
                                'reference_frames': annotated_frames,
                                'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
                                'successful_annotations': len(annotations),
                                'annotations': annotations
                            }
                            
                            # Save incrementally every 10 frames
                            if processed_count % 10 == 0:
                                save_annotations_incrementally(data, output_file)
                                print(f"  💾 Saved progress: {processed_count} new annotations")
                            
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
        
        # Save every 50 frames
        if processed_count % 50 == 0 and processed_count > 0:
            data = {
                'method': 'adaptive_polygon_prompts',
                'video_name': video_name,
                'video_path': video_path,
                'reference_frames': annotated_frames,
                'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
                'successful_annotations': len(annotations),
                'annotations': annotations
            }
            save_annotations_incrementally(data, output_file)
            print(f"  💾 Checkpoint: Saved {processed_count} new annotations")
    
    cap.release()
    
    # Final save
    data = {
        'method': 'adaptive_polygon_prompts',
        'video_name': video_name,
        'video_path': video_path,
        'reference_frames': annotated_frames,
        'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
        'successful_annotations': len(annotations),
        'annotations': annotations
    }
    
    save_annotations_incrementally(data, output_file)
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"   - New annotations: {processed_count}")
    print(f"   - Total annotations: {len(annotations)}")
    print(f"   - Reference frames used: {annotated_frames}")
    print(f"   - Saved to: {output_file}")

if __name__ == "__main__":
    print("=== IMPROVED POLYGON PROMPTS WITH ADAPTIVE REFERENCE SELECTION ===")
    print(f"Configured to process {TOTAL_FRAMES_TO_PROCESS} total frames")
    print("Uses closest reference annotation for each frame")
    print("Interpolates between references when frame is in the middle")
    test_with_improved_polygons()
