#!/usr/bin/env python3
"""
SAM2 Video Auto-labeling Script
Automatically annotates videos using SAM2 based on example frames with manual annotations.
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

# Import SAM2 components
from ultralytics.models.sam import SAM2VideoPredictor
from ultralytics import SAM

class SAM2VideoAnnotator:
    def __init__(self, 
                 videos_folder: str,
                 frames_folder: str, 
                 annotations_xml: str,
                 output_folder: str,
                 sam_model: str = "sam2_b.pt"):
        """
        Initialize the SAM2 Video Annotator
        
        Args:
            videos_folder: Path to folder containing .avi video files
            frames_folder: Path to folder containing sample frame images
            annotations_xml: Path to CVAT XML annotations file
            output_folder: Path to save output annotations
            sam_model: SAM2 model to use
        """
        self.videos_folder = Path(videos_folder)
        self.frames_folder = Path(frames_folder)
        self.annotations_xml = Path(annotations_xml)
        self.output_folder = Path(output_folder)
        self.sam_model = sam_model
        
        # Create output directory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize SAM2 models
        self.sam_image = SAM(sam_model)
        
        # Video predictor configuration
        self.predictor_config = dict(
            conf=0.25, 
            task="segment", 
            mode="predict", 
            imgsz=1024, 
            model=sam_model
        )
        
        # Parse annotations
        self.annotations = self._parse_annotations()
        self.video_examples = self._group_examples_by_video()
        
        print(f"Initialized SAM2 Video Annotator")
        print(f"Found {len(self.annotations)} annotated frames")
        print(f"Videos with examples: {list(self.video_examples.keys())}")
    
    def _parse_annotations(self) -> Dict[str, List[Dict]]:
        """Parse CVAT XML annotations"""
        tree = ET.parse(self.annotations_xml)
        root = tree.getroot()
        
        annotations = {}
        
        for image_tag in root.findall(".//image"):
            filename = image_tag.attrib['name']
            width = int(image_tag.attrib['width'])
            height = int(image_tag.attrib['height'])
            
            polygons = []
            for poly_tag in image_tag.findall(".//polygon"):
                label = poly_tag.attrib['label']
                points_str = poly_tag.attrib['points']
                
                # Parse points
                points = []
                for point_str in points_str.split(';'):
                    x, y = map(float, point_str.split(','))
                    points.append([x, y])
                
                polygons.append({
                    'label': label,
                    'points': points,
                    'bbox': self._polygon_to_bbox(points)
                })
            
            if polygons:
                annotations[filename] = {
                    'polygons': polygons,
                    'width': width,
                    'height': height
                }
        
        return annotations
    
    def _polygon_to_bbox(self, points: List[List[float]]) -> List[float]:
        """Convert polygon points to bounding box [x1, y1, x2, y2]"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def _polygon_to_center_point(self, points: List[List[float]]) -> List[float]:
        """Convert polygon to center point"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [sum(xs) / len(xs), sum(ys) / len(ys)]
    
    def _group_examples_by_video(self) -> Dict[str, List[Dict]]:
        """Group example frames by their source video"""
        video_examples = defaultdict(list)
        
        # Pattern to extract video name from frame filename
        # basler_recording_20250710_092901-frame_0000.png -> basler_recording_20250710_092901
        pattern = r'(basler_recording_\d{8}_\d{6})-frame_\d{4}\.png'
        
        for frame_name, annotation in self.annotations.items():
            match = re.match(pattern, frame_name)
            if match:
                video_name = match.group(1)
                video_examples[video_name].append({
                    'frame_name': frame_name,
                    'annotation': annotation
                })
        
        return dict(video_examples)
    
    def _get_example_prompts(self, video_name: str) -> Tuple[List[List[float]], List[int]]:
        """Get prompts (points and labels) from example frames for a video"""
        if video_name not in self.video_examples:
            return [], []
        
        all_points = []
        all_labels = []
        
        for example in self.video_examples[video_name]:
            annotation = example['annotation']
            
            for polygon_data in annotation['polygons']:
                if polygon_data['label'] == 'groove center':
                    # Use center point of polygon as prompt
                    center_point = self._polygon_to_center_point(polygon_data['points'])
                    all_points.append(center_point)
                    all_labels.append(1)  # Positive prompt
        
        return all_points, all_labels
    
    def _get_example_bboxes(self, video_name: str) -> List[List[float]]:
        """Get bounding boxes from example frames for a video"""
        if video_name not in self.video_examples:
            return []
        
        all_bboxes = []
        
        for example in self.video_examples[video_name]:
            annotation = example['annotation']
            
            for polygon_data in annotation['polygons']:
                if polygon_data['label'] == 'groove center':
                    all_bboxes.append(polygon_data['bbox'])
        
        return all_bboxes
    
    def annotate_video_with_points(self, video_path: str, output_path: str) -> bool:
        """Annotate video using point prompts from examples"""
        video_name = Path(video_path).stem
        
        # Get example prompts
        points, labels = self._get_example_prompts(video_name)
        
        if not points:
            print(f"No example prompts found for video: {video_name}")
            return False
        
        print(f"Annotating {video_name} with {len(points)} point prompts")
        
        try:
            # Create SAM2 video predictor
            predictor = SAM2VideoPredictor(overrides=self.predictor_config)
            
            # Run prediction
            if len(points) == 1:
                # Single point
                results = predictor(source=video_path, points=points[0], labels=[labels[0]])
            else:
                # Multiple points - group them as one object
                results = predictor(source=video_path, points=[points], labels=[labels])
            
            # Save results
            self._save_video_results(results, output_path, video_name)
            return True
            
        except Exception as e:
            print(f"Error annotating video {video_name}: {str(e)}")
            return False
    
    def annotate_video_with_bboxes(self, video_path: str, output_path: str) -> bool:
        """Annotate video using bounding box prompts from examples"""
        video_name = Path(video_path).stem
        
        # Get example bboxes
        bboxes = self._get_example_bboxes(video_name)
        
        if not bboxes:
            print(f"No example bboxes found for video: {video_name}")
            return False
        
        print(f"Annotating {video_name} with {len(bboxes)} bbox prompts")
        
        try:
            # For bbox prompts, we need to use the image SAM model on each frame
            # since SAM2VideoPredictor doesn't directly support bbox prompts
            return self._annotate_video_frame_by_frame(video_path, output_path, bboxes)
            
        except Exception as e:
            print(f"Error annotating video {video_name}: {str(e)}")
            return False
    
    def _annotate_video_frame_by_frame(self, video_path: str, output_path: str, bboxes: List[List[float]]) -> bool:
        """Annotate video frame by frame using bounding boxes"""
        video_name = Path(video_path).stem
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing {total_frames} frames at {fps} FPS")
        
        # Prepare output
        frame_annotations = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use first bbox as prompt (you can modify this logic)
            bbox = bboxes[0] if bboxes else None
            
            if bbox:
                try:
                    # Run SAM2 inference with bbox
                    results = self.sam_image(frame_rgb, bboxes=[bbox])
                    
                    # Extract mask and convert to polygon
                    if results and len(results) > 0:
                        masks = results[0].masks
                        if masks is not None:
                            mask = masks.data[0].cpu().numpy()
                            polygon = self._mask_to_polygon(mask)
                            
                            if polygon:
                                frame_annotations.append({
                                    'frame': frame_idx,
                                    'timestamp': frame_idx / fps,
                                    'polygon': polygon,
                                    'bbox': bbox
                                })
                
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {str(e)}")
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        
        # Save annotations
        output_file = Path(output_path) / f"{video_name}_annotations.json"
        with open(output_file, 'w') as f:
            json.dump({
                'video_name': video_name,
                'video_path': str(video_path),
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'annotations': frame_annotations
            }, f, indent=2)
        
        print(f"Saved {len(frame_annotations)} frame annotations to {output_file}")
        return True
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[List[List[float]]]:
        """Convert binary mask to polygon points"""
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of points
        points = []
        for point in simplified:
            x, y = point[0]
            points.append([float(x), float(y)])
        
        return points if len(points) >= 3 else None
    
    def _save_video_results(self, results, output_path: str, video_name: str):
        """Save video annotation results"""
        output_file = Path(output_path) / f"{video_name}_sam2_results.json"
        
        # Extract information from results
        annotations = []
        
        if results:
            for i, result in enumerate(results):
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    for j, mask in enumerate(masks):
                        polygon = self._mask_to_polygon(mask)
                        if polygon:
                            annotations.append({
                                'frame': i,
                                'mask_id': j,
                                'polygon': polygon
                            })
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump({
                'video_name': video_name,
                'total_annotations': len(annotations),
                'annotations': annotations
            }, f, indent=2)
        
        print(f"Saved {len(annotations)} annotations to {output_file}")
    
    def process_all_videos(self, method: str = "points"):
        """Process all videos in the videos folder"""
        video_files = list(self.videos_folder.glob("*.avi"))
        
        if not video_files:
            print("No .avi files found in videos folder")
            return
        
        print(f"Found {len(video_files)} videos to process")
        
        successful = 0
        failed = 0
        
        for video_file in video_files:
            print(f"\n--- Processing: {video_file.name} ---")
            
            if method == "points":
                success = self.annotate_video_with_points(str(video_file), str(self.output_folder))
            elif method == "bboxes":
                success = self.annotate_video_with_bboxes(str(video_file), str(self.output_folder))
            else:
                print(f"Unknown method: {method}")
                continue
            
            if success:
                successful += 1
            else:
                failed += 1
        
        print(f"\n=== Processing Complete ===")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")


def main():
    """Main function to run the video annotation"""
    
    # Configuration
    BASE_DIR = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    VIDEOS_FOLDER = os.path.join(BASE_DIR, "basler_recordings")
    FRAMES_FOLDER = os.path.join(BASE_DIR, "basler_recordings_frames") 
    ANNOTATIONS_XML = os.path.join(BASE_DIR, "annotations.xml")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "sam2_video_annotations")
    
    # SAM2 model - you can change this to sam2_l.pt for better accuracy
    SAM_MODEL = "sam2_b.pt"
    
    print("=== SAM2 Video Auto-labeling ===")
    print(f"Videos folder: {VIDEOS_FOLDER}")
    print(f"Frames folder: {FRAMES_FOLDER}")
    print(f"Annotations: {ANNOTATIONS_XML}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"SAM2 model: {SAM_MODEL}")
    
    # Initialize annotator
    try:
        annotator = SAM2VideoAnnotator(
            videos_folder=VIDEOS_FOLDER,
            frames_folder=FRAMES_FOLDER,
            annotations_xml=ANNOTATIONS_XML,
            output_folder=OUTPUT_FOLDER,
            sam_model=SAM_MODEL
        )
        
        # Process all videos
        # You can choose method: "points" or "bboxes"
        annotator.process_all_videos(method="bboxes")  # Using bboxes for better accuracy
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
