#!/usr/bin/env python3
"""
SAM2 Video Auto-labeling with Corrected Frame Indices

This script:
1. Uses corrected annotations with proper video frame indices
2. Performs SAM2 video segmentation starting from annotated frames
3. Propagates segmentation between annotated frames
4. Saves results with correct frame timing

Usage: Run frame_mapping_and_correction.py first to generate corrected annotations
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics.models.sam import SAM2VideoPredictor

class SAM2VideoAnnotator:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.corrected_dir = self.project_dir / "corrected_annotations"
        self.videos_dir = self.project_dir / "basler_recordings"
        self.output_dir = self.project_dir / "sam2_autolabeling_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize SAM2 Video Predictor
        self.predictor = None
        
    def initialize_sam2(self, model_path="sam2.1_l.pt"):
        """Initialize SAM2 Video Predictor"""
        print(f"Initializing SAM2 with model: {model_path}")
        
        overrides = dict(
            conf=0.25, 
            task="segment", 
            mode="predict", 
            imgsz=1024, 
            model=model_path
        )
        
        self.predictor = SAM2VideoPredictor(overrides=overrides)
        print("✅ SAM2 initialized successfully")
    
    def load_corrected_annotations(self, video_name):
        """Load corrected annotations for a specific video"""
        annotation_file = self.corrected_dir / video_name / f"{video_name}_corrected_annotations.json"
        
        if not annotation_file.exists():
            print(f"❌ Corrected annotations not found: {annotation_file}")
            print("Please run frame_mapping_and_correction.py first!")
            return None
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {data['total_annotations']} corrected annotations for {video_name}")
        return data
    
    def extract_polygon_points(self, polygon_data, num_points=8):
        """Extract representative points from polygon for SAM2 prompts"""
        points = polygon_data['points']
        
        if len(points) <= num_points:
            return points
        
        # Sample points evenly along the polygon
        step = len(points) // num_points
        sampled_points = []
        
        for i in range(0, len(points), step):
            if len(sampled_points) < num_points:
                sampled_points.append(points[i])
        
        # Add center point
        center_x = np.mean([p[0] for p in points])
        center_y = np.mean([p[1] for p in points])
        sampled_points.append([center_x, center_y])
        
        return sampled_points[:num_points]
    
    def get_frame_annotations_sorted(self, annotations_data):
        """Get annotations sorted by video frame index"""
        annotations = annotations_data['annotations']
        return sorted(annotations, key=lambda x: x['video_frame_index'])
    
    def segment_video_with_prompts(self, video_name, max_frames=None):
        """Perform video segmentation using corrected annotations as prompts"""
        print(f"\n=== Processing video: {video_name} ===")
        
        # Load corrected annotations
        annotations_data = self.load_corrected_annotations(video_name)
        if not annotations_data:
            return None
        
        # Get video path
        video_path = self.videos_dir / f"{video_name}.avi"
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return None
        
        # Get sorted annotations
        sorted_annotations = self.get_frame_annotations_sorted(annotations_data)
        
        if not sorted_annotations:
            print(f"❌ No annotations found for {video_name}")
            return None
        
        print(f"Found annotations for frames: {[ann['video_frame_index'] for ann in sorted_annotations]}")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video info: {total_frames} frames, {fps} FPS")
        
        # Create output directory for this video
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        results = {
            'video_name': video_name,
            'video_path': str(video_path),
            'total_frames': total_frames,
            'annotated_frames': [ann['video_frame_index'] for ann in sorted_annotations],
            'segments': []
        }
        
        # Process segments between annotated frames
        for i in range(len(sorted_annotations)):
            current_ann = sorted_annotations[i]
            current_frame = current_ann['video_frame_index']
            
            # Determine segment end
            if i + 1 < len(sorted_annotations):
                next_frame = sorted_annotations[i + 1]['video_frame_index']
                segment_end = next_frame
            else:
                segment_end = min(total_frames, current_frame + 100)  # Process up to 100 frames after last annotation
            
            print(f"\n--- Segment {i+1}: frames {current_frame} to {segment_end-1} ---")
            
            # Extract prompt points from current annotation
            groove_polygons = [poly for poly in current_ann['polygons'] if poly['label'] == 'groove center']
            
            if not groove_polygons:
                print(f"❌ No 'groove center' polygon found in frame {current_frame}")
                continue
            
            polygon = groove_polygons[0]
            prompt_points = self.extract_polygon_points(polygon, num_points=6)
            
            print(f"Using {len(prompt_points)} prompt points from frame {current_frame}")
            
            # Create temporary video segment for SAM2
            segment_video_path = video_output_dir / f"temp_segment_{i}.avi"
            self.extract_video_segment(video_path, current_frame, segment_end, segment_video_path)
            
            try:
                # Run SAM2 on segment
                print(f"Running SAM2 on segment...")
                results_sam2 = self.predictor(
                    source=str(segment_video_path),
                    points=prompt_points,
                    labels=[1] * len(prompt_points)  # All positive points
                )
                
                if results_sam2:
                    segment_result = self.process_sam2_results(
                        results_sam2, 
                        current_frame, 
                        segment_end,
                        video_output_dir / f"segment_{i}"
                    )
                    
                    results['segments'].append(segment_result)
                    print(f"✅ Processed segment {i+1}: {segment_result['frames_processed']} frames")
                else:
                    print(f"❌ SAM2 returned no results for segment {i+1}")
                
            except Exception as e:
                print(f"❌ Error processing segment {i+1}: {e}")
            
            finally:
                # Clean up temporary segment
                if segment_video_path.exists():
                    segment_video_path.unlink()
        
        # Save results
        results_file = video_output_dir / f"{video_name}_sam2_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Video processing complete!")
        print(f"   - Processed {len(results['segments'])} segments")
        print(f"   - Results saved to: {results_file}")
        
        return results
    
    def extract_video_segment(self, video_path, start_frame, end_frame, output_path):
        """Extract a segment from video for SAM2 processing"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Extract frames
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                out.write(frame)
            else:
                break
        
        cap.release()
        out.release()
    
    def process_sam2_results(self, results, start_frame, end_frame, output_dir):
        """Process SAM2 results and save masks/annotations"""
        output_dir.mkdir(exist_ok=True)
        
        segment_result = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames_processed': 0,
            'successful_masks': 0,
            'frame_results': []
        }
        
        frame_idx = start_frame
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for mask_idx, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()
                    
                    # Save mask as image
                    mask_image = (mask_np * 255).astype(np.uint8)
                    mask_path = output_dir / f"mask_frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(mask_path), mask_image)
                    
                    # Convert mask to polygon
                    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        polygon = []
                        for point in simplified:
                            x, y = point[0]
                            polygon.append([float(x), float(y)])
                        
                        frame_result = {
                            'frame_index': frame_idx,
                            'mask_path': str(mask_path),
                            'polygon': polygon,
                            'mask_area': int(np.sum(mask_np))
                        }
                        
                        segment_result['frame_results'].append(frame_result)
                        segment_result['successful_masks'] += 1
                
                segment_result['frames_processed'] += 1
                frame_idx += 1
        
        return segment_result
    
    def process_all_videos(self, max_frames_per_video=None):
        """Process all videos with corrected annotations"""
        print("=== SAM2 VIDEO AUTO-LABELING WITH CORRECTED ANNOTATIONS ===")
        
        if not self.predictor:
            self.initialize_sam2()
        
        # Find all videos with corrected annotations
        video_dirs = [d for d in self.corrected_dir.iterdir() if d.is_dir()]
        
        print(f"Found {len(video_dirs)} videos with corrected annotations:")
        for video_dir in video_dirs:
            print(f"  - {video_dir.name}")
        
        results = {}
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            try:
                result = self.segment_video_with_prompts(video_name, max_frames_per_video)
                if result:
                    results[video_name] = result
            except Exception as e:
                print(f"❌ Error processing {video_name}: {e}")
        
        # Save overall summary
        summary_file = self.output_dir / "autolabeling_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_videos_processed': len(results),
                'videos': list(results.keys()),
                'processing_summary': {k: {
                    'total_segments': len(v['segments']),
                    'annotated_frames': v['annotated_frames'],
                    'total_frames': v['total_frames']
                } for k, v in results.items()}
            }, f, indent=2)
        
        print(f"\n🎉 ALL VIDEOS PROCESSED!")
        print(f"Successfully processed {len(results)} videos")
        print(f"Summary saved to: {summary_file}")
        print(f"Results directory: {self.output_dir}")
        
        return results

def main():
    project_dir = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    annotator = SAM2VideoAnnotator(project_dir)
    
    # Process all videos (limit frames for testing)
    results = annotator.process_all_videos(max_frames_per_video=200)
    
    print("\n=== PROCESSING COMPLETE ===")
    for video_name, result in results.items():
        print(f"{video_name}: {len(result['segments'])} segments processed")

if __name__ == "__main__":
    main()
