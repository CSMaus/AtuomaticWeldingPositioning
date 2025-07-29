#!/usr/bin/env python3

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import SAM

def load_corrected_annotations(video_name):
    corrected_dir = Path("/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/corrected_annotations")
    json_file = corrected_dir / video_name / f"{video_name}_corrected_annotations.json"
    
    if not json_file.exists():
        print(f"No corrected annotations found for {video_name}")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def get_polygons_from_corrected_annotations(video_name):
    data = load_corrected_annotations(video_name)
    if not data:
        return {}
    
    polygons = {}
    for annotation in data['annotations']:
        frame_idx = annotation['video_frame_index']
        for polygon_data in annotation['polygons']:
            if polygon_data['label'] == 'groove center':
                polygons[frame_idx] = polygon_data['points']
                break
    
    return polygons

def get_multiple_points_from_polygon(polygon, num_points=8):
    points = []
    polygon_array = np.array(polygon)
    
    for i in range(0, len(polygon), max(1, len(polygon) // num_points)):
        points.append(polygon[i])
    
    center_x = np.mean(polygon_array[:, 0])
    center_y = np.mean(polygon_array[:, 1])
    points.append([center_x, center_y])
    
    return points

def load_existing_annotations(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        print(f"Found existing annotations file with {len(data.get('annotations', []))} annotations")
        return data
    else:
        print("No existing annotations file found, starting fresh")
        return None

def save_annotations_incrementally(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def process_video_with_corrected_annotations(video_name, max_frames=None):
    polygons = get_polygons_from_corrected_annotations(video_name)
    if not polygons:
        print(f"No polygons found for {video_name}")
        return
    
    print(f"Found {len(polygons)} annotated frames for {video_name}")
    print(f"Frame indices: {sorted(polygons.keys())}")
    
    first_frame_idx = min(polygons.keys())
    reference_polygon = polygons[first_frame_idx]
    
    print(f"Using reference polygon from frame {first_frame_idx}")
    print(f"Polygon has {len(reference_polygon)} points")
    
    prompt_points = get_multiple_points_from_polygon(reference_polygon, num_points=8)
    
    print(f"Using {len(prompt_points)} prompt points from polygon")
    
    video_path = f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/{video_name}.avi"
    output_file = f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/sam2_results/{video_name}_annotations.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    existing_data = load_existing_annotations(output_file)
    
    if existing_data:
        annotations = existing_data.get('annotations', [])
        existing_frames = {ann['frame'] for ann in annotations}
        print(f"Already have annotations for frames: {sorted(existing_frames)}")
    else:
        annotations = []
        existing_frames = set()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    data = {
        'method': 'corrected_annotations_prompts',
        'video_name': video_name,
        'video_path': video_path,
        'reference_frame': first_frame_idx,
        'reference_polygon': reference_polygon,
        'prompt_points': prompt_points,
        'total_frames_to_process': total_frames,
        'total_frames_processed': len(existing_frames),
        'successful_annotations': len(annotations),
        'corrected_annotations_used': polygons,
        'annotations': annotations
    }
    
    frames_to_process = []
    for frame_idx in range(total_frames):
        if frame_idx not in existing_frames:
            frames_to_process.append(frame_idx)
    
    if not frames_to_process:
        print(f"All {total_frames} frames already processed!")
        return
    
    print(f"Need to process {len(frames_to_process)} more frames")
    
    sam = SAM("sam2.1_l.pt")
    
    processed_count = 0
    
    for frame_idx in frames_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Cannot read frame {frame_idx}, stopping")
            break
        
        print(f"Processing frame {frame_idx}... ({processed_count + 1}/{len(frames_to_process)})")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = sam(frame_rgb, 
                         points=prompt_points, 
                         labels=[1] * len(prompt_points))
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[0].cpu().numpy()
                    
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
                                'reference_polygon': reference_polygon if frame_idx == first_frame_idx else None,
                                'prompt_points_used': prompt_points,
                                'mask_area': int(np.sum(mask)),
                                'is_reference_frame': frame_idx in polygons,
                                'original_annotation': polygons.get(frame_idx, None)
                            }
                            annotations.append(annotation)
                            processed_count += 1
                            
                            data['annotations'] = annotations
                            data['total_frames_processed'] = len(existing_frames) + processed_count
                            data['successful_annotations'] = len(annotations)
                            
                            if processed_count % 10 == 0:
                                save_annotations_incrementally(data, output_file)
                                print(f"  Saved progress: {processed_count} new annotations")
                            
                            print(f"  Frame {frame_idx}: Got polygon with {len(polygon)} points")
                        else:
                            print(f"  Frame {frame_idx}: Polygon too small")
                    else:
                        print(f"  Frame {frame_idx}: No contours")
                else:
                    print(f"  Frame {frame_idx}: No masks")
            else:
                print(f"  Frame {frame_idx}: No results")
                
        except Exception as e:
            print(f"  Frame {frame_idx}: Error - {e}")
        
        if processed_count % 50 == 0 and processed_count > 0:
            save_annotations_incrementally(data, output_file)
            print(f"  Checkpoint: Saved {processed_count} new annotations")
    
    cap.release()
    
    data['annotations'] = annotations
    data['total_frames_processed'] = len(existing_frames) + processed_count
    data['successful_annotations'] = len(annotations)
    
    save_annotations_incrementally(data, output_file)
    
    print(f"\nProcessing Complete!")
    print(f"   - New annotations: {processed_count}")
    print(f"   - Total annotations: {len(annotations)}")
    print(f"   - Saved to: {output_file}")
    print(f"   - Video: {video_name}")

def main():
    video_name = "basler_recording_20250710_092901"
    max_frames = 10
    
    print(f"=== SAM2 Processing with Corrected Annotations ===")
    print(f"Video: {video_name}")
    print(f"Max frames: {max_frames}")
    
    process_video_with_corrected_annotations(video_name, max_frames)

if __name__ == "__main__":
    main()
