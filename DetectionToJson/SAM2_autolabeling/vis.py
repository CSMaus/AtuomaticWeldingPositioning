#!/usr/bin/env python3

import cv2
import json
import numpy as np
import os
from pathlib import Path

MAX_FRAMES_TO_SHOW = 50

def load_sam2_annotations(video_name):
    json_file = f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/sam2_results/{video_name}_annotations.json"
    
    if not os.path.exists(json_file):
        print(f"Annotations not found: {json_file}")
        print("Run sam2.py first!")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded annotations from: {json_file}")
    print(f"Found {data['successful_annotations']} annotations")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Will show up to {MAX_FRAMES_TO_SHOW} frames")
    return data

def draw_polygon(frame, polygon_points, color=(0, 255, 0), thickness=2):
    if not polygon_points or len(polygon_points) < 3:
        return frame
    
    points = np.array(polygon_points, dtype=np.int32)
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    cv2.polylines(frame, [points], True, color, thickness)
    
    for point in points:
        cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
    
    return frame

def draw_reference_info(frame, frame_data, frame_idx):
    text_y = 30
    
    cv2.putText(frame, f"Frame: {frame_idx}", (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text_y += 25
    
    if frame_data.get('is_reference_frame', False):
        cv2.putText(frame, "REFERENCE FRAME", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        text_y += 25
    
    if 'total_annotations' in frame_data:
        cv2.putText(frame, f"Annotations: {frame_data['total_annotations']}", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
    
    if 'annotations' in frame_data:
        total_area = sum(ann.get('mask_area', 0) for ann in frame_data['annotations'])
        cv2.putText(frame, f"Total Mask Area: {total_area}", (10, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def visualize_sam2_annotations(video_name):
    data = load_sam2_annotations(video_name)
    if not data:
        return
    
    video_path = data.get('video_path', f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/{video_name}.avi")
    
    annotations = {ann['frame']: ann for ann in data['annotations']}
    max_frame = min(MAX_FRAMES_TO_SHOW - 1, max(annotations.keys()) if annotations else 0)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_max_frame = min(max_frame, total_video_frames - 1)
    
    print(f"\nVideo has {total_video_frames} total frames")
    print(f"Will show frames 0 to {actual_max_frame}")
    print(f"Annotations available for frames: {sorted(annotations.keys())[:10]}...")
    
    print("\n=== CONTROLS ===")
    print("'q' - Quit")
    print("'n' - Next frame")
    print("'p' - Previous frame") 
    print("SPACE - Pause/Play")
    print("'r' - Show reference frames only")
    print("'a' - Show all frames")
    
    current_frame = 0
    playing = False
    show_reference_only = False
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Cannot read frame {current_frame}")
            break
        
        frame_copy = frame.copy()
        
        if current_frame in annotations:
            frame_data = annotations[current_frame]
            
            if show_reference_only and not frame_data.get('is_reference_frame', False):
                pass
            else:
                if 'annotations' in frame_data:
                    for i, annotation in enumerate(frame_data['annotations']):
                        if 'polygon_points' in annotation:
                            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                            color = colors[i % len(colors)]
                            if frame_data.get('is_reference_frame', False):
                                color = (0, 255, 255)
                            frame_copy = draw_polygon(frame_copy, annotation['polygon_points'], color)
                
                frame_copy = draw_reference_info(frame_copy, frame_data, current_frame)
        else:
            cv2.putText(frame_copy, f"Frame: {current_frame} (No annotation)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        status_text = "PLAYING" if playing else "PAUSED"
        mode_text = "REF ONLY" if show_reference_only else "ALL FRAMES"
        cv2.putText(frame_copy, f"{status_text} | {mode_text}", (10, frame_copy.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('SAM2 Annotations Visualization', frame_copy)
        
        if playing:
            key = cv2.waitKey(100) & 0xFF
            if key == ord(' '):
                playing = False
            elif key == ord('q'):
                break
            elif key == ord('r'):
                show_reference_only = True
            elif key == ord('a'):
                show_reference_only = False
            else:
                current_frame = min(current_frame + 1, actual_max_frame)
                if current_frame >= actual_max_frame:
                    playing = False
        else:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_frame = min(current_frame + 1, actual_max_frame)
            elif key == ord('p'):
                current_frame = max(current_frame - 1, 0)
            elif key == ord(' '):
                playing = True
            elif key == ord('r'):
                show_reference_only = True
            elif key == ord('a'):
                show_reference_only = False
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nVisualization complete!")
    print(f"Showed frames 0 to {actual_max_frame}")
    print(f"Total annotations visualized: {len([f for f in range(actual_max_frame + 1) if f in annotations])}")

def main():
    video_name = "basler_recording_20250710_092901"
    
    print(f"=== SAM2 Annotations Visualization ===")
    print(f"Video: {video_name}")
    
    visualize_sam2_annotations(video_name)

if __name__ == "__main__":
    main()
