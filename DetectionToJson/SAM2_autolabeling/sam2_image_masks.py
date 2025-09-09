#!/usr/bin/env python3
"""
Use actual annotated IMAGES with their MASKS as examples for SAM2
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

def polygon_to_mask(polygon, width=1624, height=1234):
    """Convert polygon to binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def load_annotated_image_and_mask(annotation, frames_folder):
    """Load the actual annotated image and create its mask"""
    image_path = os.path.join(frames_folder, annotation['filename'])
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None, None
    
    # Load the actual annotated image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mask from polygon
    height, width = image.shape[:2]
    mask = polygon_to_mask(annotation['polygon'], width, height)
    
    return image_rgb, mask

def train_sam_with_examples(sam, all_annotations, frames_folder):
    """Train SAM2 using actual annotated images and their masks"""
    print("Training SAM2 with annotated image examples...")
    
    example_images = []
    example_masks = []
    
    for i, annotation in enumerate(all_annotations):
        image, mask = load_annotated_image_and_mask(annotation, frames_folder)
        if image is not None and mask is not None:
            example_images.append(image)
            example_masks.append(mask)
            print(f"  Loaded example {i+1}: {annotation['filename']}")
        else:
            print(f"  Failed to load: {annotation['filename']}")
    
    print(f"Successfully loaded {len(example_images)} example images with masks")
    return example_images, example_masks

def get_representative_prompts_from_examples(example_images, example_masks):
    """Get representative prompt points from the example masks"""
    all_points = []
    
    for mask in example_masks:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get center point
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                all_points.append([center_x, center_y])
            
            # Get some boundary points
            for i in range(0, len(largest_contour), max(1, len(largest_contour) // 5)):
                x, y = largest_contour[i][0]
                all_points.append([float(x), float(y)])
    
    return all_points

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
    """Process video using annotated images as examples"""
    
    video_name = "basler_recording_20250710_092901"
    video_path = f"/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings/{video_name}.avi"
    frames_folder = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/basler_recordings_frames"
    output_file = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/TEST_POLYGON_PROMPTS.json"
    
    # Get ALL annotations for this video
    all_annotations = get_all_annotations_for_video(video_name)
    if not all_annotations:
        print("No annotations found!")
        return
    
    print(f"Found {len(all_annotations)} annotated frames for {video_name}")
    
    # Initialize SAM2
    sam = SAM("sam2_b.pt")
    
    # Load annotated images and their masks as examples
    example_images, example_masks = train_sam_with_examples(sam, all_annotations, frames_folder)
    if not example_images:
        print("No example images loaded!")
        return
    
    # Get representative prompt points from examples
    prompt_points = get_representative_prompts_from_examples(example_images, example_masks)
    print(f"Generated {len(prompt_points)} prompt points from examples")
    
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
            # Use prompt points derived from example images/masks
            results = sam(frame_rgb, 
                         points=prompt_points, 
                         labels=[1] * len(prompt_points))
            
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
                                'prompt_points_used': prompt_points,
                                'example_images_count': len(example_images),
                                'method': 'image_mask_examples',
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
                'method': 'image_mask_examples',
                'video_name': video_name,
                'video_path': video_path,
                'example_images_count': len(example_images),
                'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
                'successful_annotations': len(annotations),
                'annotations': annotations
            }
            save_annotations_incrementally(data, output_file)
            print(f"  💾 Saved {processed_count} annotations")
    
    cap.release()
    
    # Final save
    data = {
        'method': 'image_mask_examples',
        'video_name': video_name,
        'video_path': video_path,
        'example_images_count': len(example_images),
        'total_frames_to_process': TOTAL_FRAMES_TO_PROCESS,
        'successful_annotations': len(annotations),
        'annotations': annotations
    }
    save_annotations_incrementally(data, output_file)
    
    print(f"\n🎉 COMPLETE!")
    print(f"Used {len(example_images)} example images with masks")
    print(f"Processed {processed_count} new frames")

if __name__ == "__main__":
    print("=== SAM2 WITH ANNOTATED IMAGES AS EXAMPLES ===")
    process_video()
