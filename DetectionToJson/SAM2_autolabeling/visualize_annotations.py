#!/usr/bin/env python3
"""
Visualize the saved annotations on video frames in real-time
Press 'q' to quit, 'n' for next frame, 'p' for previous frame, SPACE to pause/play
"""

import cv2
import json
import numpy as np
import os

# CONFIGURATION - Change this to set how many frames to show
MAX_FRAMES_TO_SHOW = 10

def load_annotations():
    """Load the saved test annotations"""
    # ONLY load the improved polygon prompts file
    json_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_POLYGON_PROMPTS.json"
    
    if not os.path.exists(json_file):
        print(f"TEST_POLYGON_PROMPTS.json not found!")
        print("Run test_with_polygon_prompt_improved.py first!")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded annotations from: TEST_POLYGON_PROMPTS.json")
    print(f"Found {data['successful_annotations']} annotations")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Will show up to {MAX_FRAMES_TO_SHOW} frames")
    return data
    
    print(f"Loaded annotations from: {json_file}")
    print(f"Found {data['successful_annotations']} annotations from {data['total_frames_processed']} frames")
    print(f"Will show up to {MAX_FRAMES_TO_SHOW} frames")
    return data

def draw_polygon(frame, polygon_points, color=(0, 255, 0), thickness=2):
    """Draw polygon on frame"""
    if not polygon_points or len(polygon_points) < 3:
        return frame
    
    # Convert to numpy array of integers
    points = np.array(polygon_points, dtype=np.int32)
    
    # Draw filled polygon (semi-transparent)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Draw polygon outline
    cv2.polylines(frame, [points], True, color, thickness)
    
    # Draw points
    for point in points:
        cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
    
    return frame

def visualize_annotations():
    """Show annotations on video frames"""
    
    # Load annotations
    data = load_annotations()
    if not data:
        return
    
    # Try different possible keys for video path
    video_path = None
    if 'test_video' in data:
        video_path = data['test_video']
    elif 'video_path' in data:
        video_path = data['video_path']
    else:
        # Default video path
        video_path = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/basler_recording_20250710_092901.avi"
        print(f"No video path in data, using default: {video_path}")
    
    annotations = {ann['frame']: ann for ann in data['annotations']}
    max_frame = MAX_FRAMES_TO_SHOW - 1
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    # Get total frames in video
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_max_frame = min(max_frame, total_video_frames - 1)
    
    print(f"\nVideo has {total_video_frames} total frames")
    print(f"Will show frames 0 to {actual_max_frame}")
    
    print("\n=== CONTROLS ===")
    print("'q' - Quit")
    print("'n' - Next frame")
    print("'p' - Previous frame") 
    print("SPACE - Pause/Play")
    print("'r' - Reset to frame 0")
    print("'j' - Jump 10 frames forward")
    print("'k' - Jump 10 frames backward")
    print("================\n")
    
    frame_idx = 0
    paused = True  # Start paused
    
    while True:
        # Always seek to current frame for precise control
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame_idx > actual_max_frame:
            print(f"Reached end at frame {frame_idx}")
            break
        
        # Create display frame
        display_frame = frame.copy()
        
        # Add frame info
        cv2.putText(display_frame, f"Frame: {frame_idx}/{actual_max_frame}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw annotation if exists
        if frame_idx in annotations:
            ann = annotations[frame_idx]
            polygon = ann['polygon_points']
            
            # Draw the groove center polygon
            display_frame = draw_polygon(display_frame, polygon, (0, 255, 0), 3)
            
            # Add annotation info
            cv2.putText(display_frame, f"Groove Center ({len(polygon)} points)", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check if mask_area exists
            if 'mask_area' in ann:
                cv2.putText(display_frame, f"Area: {ann['mask_area']} pixels", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw prompt point if exists
            if 'prompt_used' in ann:
                prompt = ann['prompt_used']
                cv2.circle(display_frame, (int(prompt[0]), int(prompt[1])), 8, (255, 0, 0), -1)
                cv2.putText(display_frame, "Prompt", (int(prompt[0])+10, int(prompt[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif 'prompt_points_used' in ann:
                # Multiple prompts
                for i, prompt in enumerate(ann['prompt_points_used']):
                    cv2.circle(display_frame, (int(prompt[0]), int(prompt[1])), 6, (255, 0, 0), -1)
                    cv2.putText(display_frame, str(i+1), (int(prompt[0])+8, int(prompt[1])-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        else:
            cv2.putText(display_frame, "No annotation for this frame", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add pause indicator
        if paused:
            cv2.putText(display_frame, "PAUSED", (display_frame.shape[1]-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('SAM2 Annotations Visualization', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(100 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('n'):  # Next frame
            if frame_idx < actual_max_frame:
                frame_idx += 1
        elif key == ord('p'):  # Previous frame
            if frame_idx > 0:
                frame_idx -= 1
        elif key == ord(' '):  # Space - pause/play
            paused = not paused
        elif key == ord('r'):  # Reset
            frame_idx = 0
        elif key == ord('j'):  # Jump forward 10 frames
            frame_idx = min(frame_idx + 10, actual_max_frame)
        elif key == ord('k'):  # Jump backward 10 frames
            frame_idx = max(frame_idx - 10, 0)
        
        # Auto advance if not paused
        if not paused:
            frame_idx += 1
            if frame_idx > actual_max_frame:
                print("Reached end, pausing...")
                paused = True
                frame_idx = actual_max_frame

    cap.release()
    cv2.destroyAllWindows()
    
    print("\nVisualization complete!")
    print(f"Showed annotations for {len(annotations)} frames")

if __name__ == "__main__":
    print("=== SAM2 ANNOTATIONS VISUALIZER ===")
    print(f"Configured to show up to {MAX_FRAMES_TO_SHOW} frames")
    print("Change MAX_FRAMES_TO_SHOW variable at the top of the script to adjust")
    visualize_annotations()
