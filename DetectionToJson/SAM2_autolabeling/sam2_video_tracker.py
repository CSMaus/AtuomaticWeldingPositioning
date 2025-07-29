#!/usr/bin/env python3
"""
SAM2 Video Tracker - Simplified version
Uses SAM2VideoPredictor to track objects across video frames using example prompts.
"""

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import cv2

from ultralytics.models.sam import SAM2VideoPredictor

class SAM2VideoTracker:
    def __init__(self, 
                 videos_folder: str,
                 annotations_xml: str,
                 output_folder: str,
                 sam_model: str = "sam2_b.pt"):
        """
        Initialize SAM2 Video Tracker
        
        Args:
            videos_folder: Path to folder containing .avi video files
            annotations_xml: Path to CVAT XML annotations file
            output_folder: Path to save output annotations
            sam_model: SAM2 model to use
        """
        self.videos_folder = Path(videos_folder)
        self.annotations_xml = Path(annotations_xml)
        self.output_folder = Path(output_folder)
        self.sam_model = sam_model
        
        # Create output directory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Parse annotations and group by video
        self.annotations = self._parse_annotations()
        self.video_prompts = self._extract_video_prompts()
        
        print(f"Initialized SAM2 Video Tracker")
        print(f"Found annotations for {len(self.video_prompts)} videos")
        for video_name, prompts in self.video_prompts.items():
            print(f"  {video_name}: {len(prompts['points'])} prompts")
    
    def _parse_annotations(self) -> Dict[str, Dict]:
        """Parse CVAT XML annotations"""
        tree = ET.parse(self.annotations_xml)
        root = tree.getroot()
        
        annotations = {}
        
        for image_tag in root.findall(".//image"):
            filename = image_tag.attrib['name']
            
            # Extract groove center polygons
            groove_polygons = []
            for poly_tag in image_tag.findall(".//polygon"):
                if poly_tag.attrib['label'] == 'groove center':
                    points_str = poly_tag.attrib['points']
                    points = []
                    for point_str in points_str.split(';'):
                        x, y = map(float, point_str.split(','))
                        points.append([x, y])
                    groove_polygons.append(points)
            
            if groove_polygons:
                annotations[filename] = groove_polygons
        
        return annotations
    
    def _extract_video_prompts(self) -> Dict[str, Dict]:
        """Extract prompts for each video from example frames"""
        video_prompts = defaultdict(lambda: {'points': [], 'labels': []})
        
        # Pattern to extract video name from frame filename
        pattern = r'(basler_recording_\d{8}_\d{6})-frame_\d{4}\.png'
        
        for frame_name, polygons in self.annotations.items():
            match = re.match(pattern, frame_name)
            if match:
                video_name = match.group(1)
                
                # Convert each polygon to center point
                for polygon in polygons:
                    center_x = sum(p[0] for p in polygon) / len(polygon)
                    center_y = sum(p[1] for p in polygon) / len(polygon)
                    
                    video_prompts[video_name]['points'].append([center_x, center_y])
                    video_prompts[video_name]['labels'].append(1)  # Positive prompt
        
        return dict(video_prompts)
    
    def track_video(self, video_path: str) -> bool:
        """Track objects in a single video using SAM2"""
        video_name = Path(video_path).stem
        
        # Check if video was already processed
        output_file = self.output_folder / f"{video_name}_tracking.json"
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            if existing_data.get('results_summary', {}).get('processing_status') == 'completed':
                print(f"✅ Video {video_name} already completed. Skipping...")
                return True
            elif existing_data.get('results_summary', {}).get('processing_status') == 'in_progress':
                last_frame = existing_data.get('results_summary', {}).get('last_processed_frame', -1)
                print(f"⚠️  Video {video_name} was partially processed (up to frame {last_frame})")
                print(f"   Restarting from beginning...")
        
        if video_name not in self.video_prompts:
            print(f"❌ No prompts found for video: {video_name}")
            return False
        
        prompts = self.video_prompts[video_name]
        points = prompts['points']
        labels = prompts['labels']
        
        print(f"🎯 Tracking {video_name} with {len(points)} prompts...")
        
        try:
            # Configure SAM2 video predictor
            overrides = dict(
                conf=0.25, 
                task="segment", 
                mode="predict", 
                imgsz=1024, 
                model=self.sam_model,
                save=True,  # Save visualization video
                project=str(self.output_folder),
                name=video_name,
                exist_ok=True  # Allow overwriting
            )
            
            predictor = SAM2VideoPredictor(overrides=overrides)
            
            # Run tracking
            if len(points) == 1:
                # Single point tracking
                results = predictor(
                    source=str(video_path), 
                    points=points[0], 
                    labels=[labels[0]]
                )
            else:
                # Multiple points - track as separate objects
                results = predictor(
                    source=str(video_path), 
                    points=points, 
                    labels=labels
                )
            
            # Save tracking results incrementally
            self._save_tracking_results(results, video_name, points, labels)
            
            print(f"✅ Successfully completed tracking for: {video_name}")
            return True
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Tracking interrupted for {video_name}")
            print(f"   Partial results may be saved in: {output_file}")
            return False
        except Exception as e:
            print(f"❌ Error tracking video {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_tracking_results(self, results, video_name: str, points: List, labels: List):
        """Save tracking results incrementally as frames are processed"""
        output_file = self.output_folder / f"{video_name}_tracking.json"
        yolo_folder = self.output_folder / "yolo_annotations"
        yolo_folder.mkdir(exist_ok=True)
        
        # Initialize tracking data structure
        tracking_data = {
            'video_name': video_name,
            'prompts': {
                'points': points,
                'labels': labels
            },
            'results_summary': {
                'total_results': len(results) if results else 0,
                'tracking_successful': results is not None and len(results) > 0,
                'processing_status': 'in_progress'
            },
            'frame_results': []
        }
        
        # Save initial file
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Initialized tracking file: {output_file}")
        
        # Process frames incrementally
        if results:
            print(f"Processing {len(results)} frames with incremental saving...")
            
            for i, result in enumerate(results):
                frame_data = {
                    'frame_index': i,
                    'has_masks': hasattr(result, 'masks') and result.masks is not None,
                    'processed_at': self._get_timestamp()
                }
                
                if frame_data['has_masks']:
                    masks = result.masks
                    frame_data['num_masks'] = len(masks.data) if masks.data is not None else 0
                    
                    # Extract mask statistics and polygon
                    if masks.data is not None and len(masks.data) > 0:
                        mask = masks.data[0].cpu().numpy()
                        frame_data['mask_area'] = int(np.sum(mask))
                        frame_data['mask_bbox'] = self._get_mask_bbox(mask)
                        
                        # Convert mask to polygon
                        polygon = self._mask_to_polygon(mask)
                        if polygon:
                            frame_data['polygon'] = polygon
                            
                            # Save YOLO format immediately
                            self._save_single_yolo_annotation(polygon, video_name, i, yolo_folder)
                        
                        # Add confidence if available
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            if hasattr(result.boxes, 'conf'):
                                frame_data['confidence'] = float(result.boxes.conf[0]) if len(result.boxes.conf) > 0 else None
                
                # Add frame to tracking data
                tracking_data['frame_results'].append(frame_data)
                
                # Save updated JSON every 10 frames or on last frame
                if (i + 1) % 10 == 0 or i == len(results) - 1:
                    tracking_data['results_summary']['last_processed_frame'] = i
                    tracking_data['results_summary']['processing_status'] = 'completed' if i == len(results) - 1 else 'in_progress'
                    
                    with open(output_file, 'w') as f:
                        json.dump(tracking_data, f, indent=2)
                    
                    print(f"Saved progress: {i + 1}/{len(results)} frames processed")
        
        # Final save
        tracking_data['results_summary']['processing_status'] = 'completed'
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        frames_with_polygons = len([f for f in tracking_data['frame_results'] if 'polygon' in f])
        print(f"✅ Completed tracking for {video_name}")
        print(f"   - Total frames: {len(tracking_data['frame_results'])}")
        print(f"   - Frames with polygons: {frames_with_polygons}")
        print(f"   - JSON saved to: {output_file}")
        print(f"   - YOLO files saved to: {yolo_folder}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_single_yolo_annotation(self, polygon: List[List[float]], video_name: str, frame_idx: int, yolo_folder: Path):
        """Save a single frame's annotation in YOLO format immediately"""
        yolo_file = yolo_folder / f"{video_name}_frame_{frame_idx:04d}.txt"
        
        with open(yolo_file, 'w') as f:
            # Class 0 for groove center, followed by polygon points
            points_str = " ".join([f"{p[0]:.2f} {p[1]:.2f}" for p in polygon])
            f.write(f"0 {points_str}\n")
    
    def _get_mask_bbox(self, mask: np.ndarray) -> List[int]:
        """Get bounding box from binary mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [int(cmin), int(rmin), int(cmax), int(rmax)]
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """Convert binary mask to polygon points"""
        import cv2
        
        # Ensure mask is uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce points
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of points
        points = []
        for point in simplified:
            x, y = point[0]
            points.append([float(x), float(y)])
        
        return points if len(points) >= 3 else []
    
    def track_all_videos(self):
        """Track all videos in the videos folder"""
        video_files = list(self.videos_folder.glob("*.avi"))
        
        if not video_files:
            print("❌ No .avi files found in videos folder")
            return
        
        print(f"🎬 Found {len(video_files)} videos to track")
        
        successful = 0
        failed = 0
        skipped = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n--- [{i}/{len(video_files)}] Processing: {video_file.name} ---")
            
            success = self.track_video(str(video_file))
            
            if success:
                successful += 1
            else:
                failed += 1
        
        print(f"\n=== 🏁 Tracking Complete ===")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Results saved in: {self.output_folder}")


def main():
    """Main function"""
    
    # Configuration
    BASE_DIR = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    VIDEOS_FOLDER = os.path.join(BASE_DIR, "basler_recordings")
    ANNOTATIONS_XML = os.path.join(BASE_DIR, "annotations.xml")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "sam2_tracking_results")
    
    # SAM2 model options: sam2_t.pt, sam2_s.pt, sam2_b.pt, sam2_l.pt
    SAM_MODEL = "sam2_b.pt"  # Base model - good balance of speed/accuracy
    
    print("=== SAM2 Video Tracking ===")
    print(f"Videos folder: {VIDEOS_FOLDER}")
    print(f"Annotations: {ANNOTATIONS_XML}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"SAM2 model: {SAM_MODEL}")
    
    # Initialize tracker
    try:
        tracker = SAM2VideoTracker(
            videos_folder=VIDEOS_FOLDER,
            annotations_xml=ANNOTATIONS_XML,
            output_folder=OUTPUT_FOLDER,
            sam_model=SAM_MODEL
        )
        
        # Track all videos
        tracker.track_all_videos()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
