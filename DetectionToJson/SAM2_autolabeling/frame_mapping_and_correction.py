#!/usr/bin/env python3

import os
import cv2
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

class FrameMapper:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.frames_dir = self.project_dir / "basler_recordings_frames"
        self.videos_dir = self.project_dir / "basler_recordings"
        self.annotations_file = self.project_dir / "annotations.xml"
        self.output_dir = self.project_dir / "corrected_annotations"
        
    def extract_video_name_from_image(self, image_filename):
        return image_filename.split('-frame_')[0]
    
    def extract_image_frame_number(self, image_filename):
        frame_part = image_filename.split('-frame_')[1].split('.')[0]
        return int(frame_part)
    
    def find_exact_matching_video_frame_optimized(self, video_path, target_image):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Searching through {total_frames} video frames for best match...")
        
        target_img = cv2.imread(str(target_image))
        if target_img is None:
            print(f"Cannot load target image: {target_image}")
            cap.release()
            return None
        
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_height, target_width = target_img.shape[:2]
        
        best_match_frame = None
        best_similarity = 0.0
        
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_height, frame_width = frame.shape[:2]
            if frame_height != target_height or frame_width != target_width:
                continue
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            result = cv2.matchTemplate(target_gray, frame_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_similarity:
                best_similarity = max_val
                best_match_frame = frame_idx
            
            if max_val > 0.987:
                print(f"High match found at video frame {frame_idx} with similarity {max_val:.4f}")
                cap.release()
                return frame_idx
            
            if frame_idx % 50 == 0:
                print(f"  Checked frame {frame_idx}/{total_frames}, best: {best_similarity:.4f}")
        
        cap.release()
        
        if best_match_frame is not None:
            print(f"Best match found at video frame {best_match_frame} with similarity {best_similarity:.4f}")
            return best_match_frame
        else:
            print(f"No good match found for {target_image.name}")
            return None
    
    def load_annotations_from_xml(self):
        tree = ET.parse(self.annotations_file)
        root = tree.getroot()
        
        annotations = {}
        for image_tag in root.findall(".//image"):
            image_id = int(image_tag.attrib['id'])
            filename = image_tag.attrib['name']
            width = int(image_tag.attrib['width'])
            height = int(image_tag.attrib['height'])
            
            polygons = []
            for poly_tag in image_tag.findall(".//polygon"):
                label = poly_tag.attrib['label']
                points_str = poly_tag.attrib['points']
                
                polygon_points = []
                for point_str in points_str.split(';'):
                    x, y = map(float, point_str.split(','))
                    polygon_points.append([x, y])
                
                polygons.append({
                    'label': label,
                    'points': polygon_points,
                    'source': poly_tag.attrib.get('source', 'manual'),
                    'occluded': poly_tag.attrib.get('occluded', '0'),
                    'z_order': poly_tag.attrib.get('z_order', '0')
                })
            
            annotations[filename] = {
                'id': image_id,
                'filename': filename,
                'width': width,
                'height': height,
                'polygons': polygons
            }
        
        return annotations
    
    def create_frame_mapping(self, video_name):
        print(f"\n=== Creating frame mapping for {video_name} ===")
        
        video_path = self.videos_dir / f"{video_name}.avi"
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return None
        
        image_files = []
        for img_file in self.frames_dir.glob(f"{video_name}-frame_*.png"):
            image_files.append(img_file)
        
        if not image_files:
            print(f"No image frames found for video: {video_name}")
            return None
        
        image_files.sort(key=lambda x: self.extract_image_frame_number(x.name))
        
        print(f"Found {len(image_files)} image frames for {video_name}")
        
        frame_mapping = {}
        
        for img_file in image_files:
            image_frame_num = self.extract_image_frame_number(img_file.name)
            print(f"\nMapping image frame {image_frame_num} ({img_file.name})...")
            
            video_frame_idx = self.find_exact_matching_video_frame_optimized(video_path, img_file)
            
            if video_frame_idx is not None:
                frame_mapping[img_file.name] = {
                    'image_frame_number': image_frame_num,
                    'video_frame_index': video_frame_idx,
                    'image_path': str(img_file)
                }
                print(f"  Mapped to video frame {video_frame_idx}")
            else:
                print(f"  Could not find matching video frame")
        
        return frame_mapping
    
    def create_corrected_annotations(self, video_name):
        print(f"\n=== Processing {video_name} ===")
        
        frame_mapping = self.create_frame_mapping(video_name)
        if not frame_mapping:
            return None
        
        original_annotations = self.load_annotations_from_xml()
        
        video_output_dir = self.output_dir / video_name
        frames_output_dir = video_output_dir / "frames"
        frames_output_dir.mkdir(parents=True, exist_ok=True)
        
        corrected_annotations = []
        
        for image_filename, mapping in frame_mapping.items():
            if image_filename in original_annotations:
                annotation = original_annotations[image_filename]
                video_frame_idx = mapping['video_frame_index']
                
                src_image = Path(mapping['image_path'])
                dst_image = frames_output_dir / f"{video_name}-corrected_frame_{video_frame_idx:04d}.png"
                shutil.copy2(src_image, dst_image)
                
                corrected_annotation = {
                    'original_image_id': annotation['id'],
                    'original_filename': annotation['filename'],
                    'corrected_filename': dst_image.name,
                    'video_frame_index': video_frame_idx,
                    'image_frame_number': mapping['image_frame_number'],
                    'width': annotation['width'],
                    'height': annotation['height'],
                    'polygons': annotation['polygons']
                }
                
                corrected_annotations.append(corrected_annotation)
                
                print(f"  {image_filename} -> frame {video_frame_idx:04d}")
        
        corrected_annotations.sort(key=lambda x: x['video_frame_index'])
        
        json_output = video_output_dir / f"{video_name}_corrected_annotations.json"
        with open(json_output, 'w') as f:
            json.dump({
                'video_name': video_name,
                'total_annotations': len(corrected_annotations),
                'frame_mapping': frame_mapping,
                'annotations': corrected_annotations
            }, f, indent=2)
        
        xml_output = video_output_dir / f"{video_name}_corrected_annotations.xml"
        self.create_corrected_xml(corrected_annotations, xml_output, video_name)
        
        print(f"\nCreated corrected annotations for {video_name}:")
        print(f"   - Frames: {frames_output_dir}")
        print(f"   - JSON: {json_output}")
        print(f"   - XML: {xml_output}")
        print(f"   - Total annotations: {len(corrected_annotations)}")
        
        return {
            'video_name': video_name,
            'frame_mapping': frame_mapping,
            'corrected_annotations': corrected_annotations,
            'output_dir': video_output_dir
        }
    
    def create_corrected_xml(self, corrected_annotations, output_path, video_name):
        root = ET.Element("annotations")
        
        version = ET.SubElement(root, "version")
        version.text = "1.1"
        
        meta = ET.SubElement(root, "meta")
        job = ET.SubElement(meta, "job")
        
        job_id = ET.SubElement(job, "id")
        job_id.text = "corrected_" + video_name
        
        size = ET.SubElement(job, "size")
        size.text = str(len(corrected_annotations))
        
        mode = ET.SubElement(job, "mode")
        mode.text = "annotation"
        
        start_frame = ET.SubElement(job, "start_frame")
        start_frame.text = "0"
        
        stop_frame = ET.SubElement(job, "stop_frame")
        stop_frame.text = str(len(corrected_annotations) - 1)
        
        labels = ET.SubElement(job, "labels")
        
        unique_labels = set()
        for ann in corrected_annotations:
            for poly in ann['polygons']:
                unique_labels.add(poly['label'])
        
        for label_name in unique_labels:
            label = ET.SubElement(labels, "label")
            n = ET.SubElement(label, "n")
            n.text = label_name
            color = ET.SubElement(label, "color")
            color.text = "#fa3253"
            label_type = ET.SubElement(label, "type")
            label_type.text = "any"
        
        for i, ann in enumerate(corrected_annotations):
            image = ET.SubElement(root, "image")
            image.set("id", str(i))
            image.set("name", ann['corrected_filename'])
            image.set("width", str(ann['width']))
            image.set("height", str(ann['height']))
            
            for poly in ann['polygons']:
                polygon = ET.SubElement(image, "polygon")
                polygon.set("label", poly['label'])
                polygon.set("source", poly['source'])
                polygon.set("occluded", poly['occluded'])
                polygon.set("z_order", poly['z_order'])
                
                points_str = ";".join([f"{p[0]},{p[1]}" for p in poly['points']])
                polygon.set("points", points_str)
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def process_all_videos(self):
        print("=== FRAME MAPPING AND ANNOTATION CORRECTION ===")
        
        video_names = set()
        for img_file in self.frames_dir.glob("*.png"):
            video_name = self.extract_video_name_from_image(img_file.name)
            video_names.add(video_name)
        
        print(f"Found {len(video_names)} unique videos:")
        for video_name in sorted(video_names):
            print(f"  - {video_name}")
        
        results = {}
        
        for video_name in sorted(video_names):
            try:
                result = self.create_corrected_annotations(video_name)
                if result:
                    results[video_name] = result
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
        
        print(f"\nPROCESSING COMPLETE!")
        print(f"Successfully processed {len(results)} videos")
        print(f"Output directory: {self.output_dir}")
        
        return results

def main():
    project_dir = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    mapper = FrameMapper(project_dir)
    results = mapper.process_all_videos()
    
    summary_file = mapper.output_dir / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_videos_processed': len(results),
            'videos': list(results.keys()),
            'processing_results': {k: {
                'total_annotations': len(v['corrected_annotations']),
                'output_dir': str(v['output_dir'])
            } for k, v in results.items()}
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
