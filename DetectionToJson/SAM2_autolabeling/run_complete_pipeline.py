#!/usr/bin/env python3
"""
Complete SAM2 Video Annotation Pipeline

This script runs the complete pipeline:
1. Frame mapping and annotation correction
2. SAM2 video auto-labeling with corrected annotations

Usage: python run_complete_pipeline.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from frame_mapping_and_correction import FrameMapper
from sam2_video_autolabeling_corrected import SAM2VideoAnnotator

def main():
    project_dir = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    print("🚀 STARTING COMPLETE SAM2 VIDEO ANNOTATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Frame mapping and correction
    print("\n📍 STEP 1: Frame Mapping and Annotation Correction")
    print("-" * 50)
    
    try:
        mapper = FrameMapper(project_dir)
        mapping_results = mapper.process_all_videos()
        
        if not mapping_results:
            print("❌ No videos were successfully mapped. Stopping pipeline.")
            return
        
        print(f"✅ Step 1 complete: {len(mapping_results)} videos mapped")
        
    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        return
    
    # Step 2: SAM2 video auto-labeling
    print("\n🎯 STEP 2: SAM2 Video Auto-labeling")
    print("-" * 50)
    
    try:
        annotator = SAM2VideoAnnotator(project_dir)
        annotation_results = annotator.process_all_videos(max_frames_per_video=100)  # Limit for testing
        
        if not annotation_results:
            print("❌ No videos were successfully annotated.")
            return
        
        print(f"✅ Step 2 complete: {len(annotation_results)} videos annotated")
        
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        return
    
    # Summary
    print("\n🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"✅ Frame mapping: {len(mapping_results)} videos")
    print(f"✅ Auto-labeling: {len(annotation_results)} videos")
    
    print("\nOutput directories:")
    print(f"  📁 Corrected annotations: {project_dir}/corrected_annotations/")
    print(f"  📁 SAM2 results: {project_dir}/sam2_autolabeling_results/")
    
    print("\nNext steps:")
    print("  1. Review the corrected frame mappings")
    print("  2. Check SAM2 segmentation results")
    print("  3. Adjust parameters if needed and re-run")

if __name__ == "__main__":
    main()
