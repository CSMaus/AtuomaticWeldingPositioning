import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import cv2
from datetime import datetime

from ultralytics.models.sam import SAM2VideoPredictor

class SAM2CorrectedTracker:
    def __init__(self, 
                 corrected_annotations_folder: str,
                 videos_folder: str,
                 output_folder: str,
                 sam_model: str = "sam2_b.pt"):
        
        self.corrected_annotations_folder = Path(corrected_annotations_folder)
        self.videos_folder = Path(videos_folder)
        self.output_folder = Path(output_folder)
        self.sam_model = sam_model
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.video_data = self._load_corrected_annotations()
        self.video_prompts = self._extract_video_prompts()
        
        print(f"Initialized SAM2 Corrected Tracker")
        print(f"Found annotations for {len(self.video_prompts)} videos")
        for video_name, prompts in self.video_prompts.items():
            print(f"  {video_name}: {len(prompts['points'])} prompts")
    
    def _load_corrected_annotations(self) -> Dict[str, Dict]:
        video_data = {}
        
        for video_dir in self.corrected_annotations_folder.iterdir():
            if video_dir.is_dir():
                video_name = video_dir.name
                json_file = video_dir / f"{video_name}_corrected_annotations.json"
                
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    video_data[video_name] = data
        
        return video_data
    
    def _extract_video_prompts(self) -> Dict[str, Dict]:
        video_prompts = {}
        
        for video_name, video_data in self.video_data.items():
            points = []
            labels = []
            
            for annotation in video_data['annotations']:
                for polygon_data in annotation['polygons']:
                    if polygon_data['label'] == 'groove center':
                        polygon_points = polygon_data['points']
                        
                        center_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                        center_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                        
                        points.append([center_x, center_y])
                        labels.append(1)
            
            if points:
                video_prompts[video_name] = {'points': points, 'labels': labels}
        
        return video_prompts
    
    def track_video(self, video_path: str) -> bool:
        video_name = Path(video_path).stem
        
        output_file = self.output_folder / f"{video_name}_tracking.json"
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            if existing_data.get('results_summary', {}).get('processing_status') == 'completed':
                print(f"Video {video_name} already completed. Skipping...")
                return True
            elif existing_data.get('results_summary', {}).get('processing_status') == 'in_progress':
                last_frame = existing_data.get('results_summary', {}).get('last_processed_frame', -1)
                print(f"Video {video_name} was partially processed (up to frame {last_frame})")
                print(f"Restarting from beginning...")
        
        if video_name not in self.video_prompts:
            print(f"No prompts found for video: {video_name}")
            return False
        
        prompts = self.video_prompts[video_name]
        points = prompts['points']
        labels = prompts['labels']
        
        print(f"Tracking {video_name} with {len(points)} prompts...")
        
        try:
            overrides = dict(
                conf=0.25, 
                task="segment", 
                mode="predict", 
                imgsz=1024, 
                model=self.sam_model,
                save=True,
                project=str(self.output_folder),
                name=video_name,
                exist_ok=True
            )
            
            predictor = SAM2VideoPredictor(overrides=overrides)
            
            if len(points) == 1:
                results = predictor(
                    source=str(video_path), 
                    points=points[0], 
                    labels=[labels[0]]
                )
            else:
                results = predictor(
                    source=str(video_path), 
                    points=points, 
                    labels=labels
                )
            
            self._save_tracking_results(results, video_name, points, labels)
            
            print(f"Successfully completed tracking for: {video_name}")
            return True
            
        except KeyboardInterrupt:
            print(f"\nTracking interrupted for {video_name}")
            print(f"Partial results may be saved in: {output_file}")
            return False
        except Exception as e:
            print(f"Error tracking video {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_tracking_results(self, results, video_name: str, points: List, labels: List):
        output_file = self.output_folder / f"{video_name}_tracking.json"
        json_annotations_folder = self.output_folder / "json_annotations"
        json_annotations_folder.mkdir(exist_ok=True)
        
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
        
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Initialized tracking file: {output_file}")
        
        if results:
            print(f"Processing {len(results)} frames with incremental saving...")
            
            for i, result in enumerate(results):
                frame_data = {
                    'frame_index': i,
                    'has_masks': hasattr(result, 'masks') and result.masks is not None,
                    'processed_at': self._get_timestamp(),
                    'annotations': []
                }
                
                if frame_data['has_masks']:
                    masks = result.masks
                    frame_data['num_masks'] = len(masks.data) if masks.data is not None else 0
                    
                    if masks.data is not None and len(masks.data) > 0:
                        for mask_idx, mask in enumerate(masks.data):
                            mask_np = mask.cpu().numpy()
                            
                            polygon = self._mask_to_polygon(mask_np)
                            bbox = self._get_mask_bbox(mask_np)
                            
                            annotation = {
                                'mask_index': mask_idx,
                                'polygon': polygon,
                                'bbox': bbox,
                                'mask_area': int(np.sum(mask_np)),
                                'polygon_area': self._calculate_polygon_area(polygon) if polygon else 0,
                                'polygon_points_count': len(polygon) if polygon else 0
                            }
                            
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                if hasattr(result.boxes, 'conf') and len(result.boxes.conf) > mask_idx:
                                    annotation['confidence'] = float(result.boxes.conf[mask_idx])
                            
                            frame_data['annotations'].append(annotation)
                
                tracking_data['frame_results'].append(frame_data)
                
                frame_json_file = json_annotations_folder / f"{video_name}_frame_{i:04d}.json"
                with open(frame_json_file, 'w') as f:
                    json.dump(frame_data, f, indent=2)
                
                if (i + 1) % 10 == 0 or i == len(results) - 1:
                    tracking_data['results_summary']['last_processed_frame'] = i
                    tracking_data['results_summary']['processing_status'] = 'completed' if i == len(results) - 1 else 'in_progress'
                    
                    with open(output_file, 'w') as f:
                        json.dump(tracking_data, f, indent=2)
                    
                    print(f"Saved progress: {i + 1}/{len(results)} frames processed")
        
        tracking_data['results_summary']['processing_status'] = 'completed'
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        frames_with_annotations = len([f for f in tracking_data['frame_results'] if f['annotations']])
        print(f"Completed tracking for {video_name}")
        print(f"   - Total frames: {len(tracking_data['frame_results'])}")
        print(f"   - Frames with annotations: {frames_with_annotations}")
        print(f"   - JSON saved to: {output_file}")
        print(f"   - Frame JSONs saved to: {json_annotations_folder}")
    
    def _get_timestamp(self):
        return datetime.now().isoformat()
    
    def _get_mask_bbox(self, mask: np.ndarray) -> List[int]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [int(cmin), int(rmin), int(cmax), int(rmax)]
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        points = []
        for point in simplified:
            x, y = point[0]
            points.append([float(x), float(y)])
        
        return points if len(points) >= 3 else []
    
    def _calculate_polygon_area(self, polygon_points: List[List[float]]) -> float:
        if len(polygon_points) < 3:
            return 0.0
        
        area = 0.0
        n = len(polygon_points)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon_points[i][0] * polygon_points[j][1]
            area -= polygon_points[j][0] * polygon_points[i][1]
        
        return abs(area) / 2.0
    
    def track_all_videos(self):
        video_files = list(self.videos_folder.glob("*.avi"))
        
        if not video_files:
            print("No .avi files found in videos folder")
            return
        
        print(f"Found {len(video_files)} videos to track")
        
        successful = 0
        failed = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n--- [{i}/{len(video_files)}] Processing: {video_file.name} ---")
            
            success = self.track_video(str(video_file))
            
            if success:
                successful += 1
            else:
                failed += 1
        
        print(f"\n=== Tracking Complete ===")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved in: {self.output_folder}")

def main():
    BASE_DIR = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    CORRECTED_ANNOTATIONS_FOLDER = os.path.join(BASE_DIR, "corrected_annotations")
    VIDEOS_FOLDER = os.path.join(BASE_DIR, "basler_recordings")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "sam2_corrected_results")
    
    SAM_MODEL = "sam2_b.pt"
    
    print("=== SAM2 Corrected Video Tracking ===")
    print(f"Corrected annotations: {CORRECTED_ANNOTATIONS_FOLDER}")
    print(f"Videos folder: {VIDEOS_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"SAM2 model: {SAM_MODEL}")
    
    try:
        tracker = SAM2CorrectedTracker(
            corrected_annotations_folder=CORRECTED_ANNOTATIONS_FOLDER,
            videos_folder=VIDEOS_FOLDER,
            output_folder=OUTPUT_FOLDER,
            sam_model=SAM_MODEL
        )
        
        tracker.track_all_videos()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
