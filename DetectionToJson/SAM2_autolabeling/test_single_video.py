#!/usr/bin/env python3
"""
Test Script for Single Video Processing

This script tests the pipeline on a single video to verify everything works correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from frame_mapping_and_correction import FrameMapper
from sam2_video_autolabeling_corrected import SAM2VideoAnnotator

def test_single_video(video_name="basler_recording_20250710_092901"):
    """Test the pipeline on a single video"""
    project_dir = "/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling"
    
    print(f"🧪 TESTING PIPELINE ON: {video_name}")
    print("=" * 60)
    
    # Step 1: Frame mapping for single video
    print("\n📍 STEP 1: Frame Mapping and Correction")
    print("-" * 50)
    
    try:
        mapper = FrameMapper(project_dir)
        result = mapper.create_corrected_annotations(video_name)
        
        if not result:
            print(f"❌ Failed to map frames for {video_name}")
            return
        
        print(f"✅ Step 1 complete: {len(result['corrected_annotations'])} annotations mapped")
        
    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        return
    
    # Step 2: SAM2 annotation for single video
    print("\n🎯 STEP 2: SAM2 Video Auto-labeling")
    print("-" * 50)
    
    try:
        annotator = SAM2VideoAnnotator(project_dir)
        annotator.initialize_sam2()
        
        sam2_result = annotator.segment_video_with_prompts(video_name, max_frames=50)  # Test with 50 frames
        
        if not sam2_result:
            print(f"❌ Failed to annotate {video_name}")
            return
        
        print(f"✅ Step 2 complete: {len(sam2_result['segments'])} segments processed")
        
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        return
    
    # Summary
    print(f"\n🎉 TEST COMPLETE FOR {video_name}!")
    print("=" * 60)
    print(f"✅ Mapped annotations: {len(result['corrected_annotations'])}")
    print(f"✅ SAM2 segments: {len(sam2_result['segments'])}")
    
    print(f"\nCheck results in:")
    print(f"  📁 {project_dir}/corrected_annotations/{video_name}/")
    print(f"  📁 {project_dir}/sam2_autolabeling_results/{video_name}/")

def list_available_videos():
    """List all available videos for testing"""
    project_dir = Path("/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling")
    frames_dir = project_dir / "basler_recordings_frames"
    
    video_names = set()
    for img_file in frames_dir.glob("*.png"):
        video_name = img_file.name.split('-frame_')[0]
        video_names.add(video_name)
    
    print("Available videos for testing:")
    for i, video_name in enumerate(sorted(video_names), 1):
        print(f"  {i}. {video_name}")
    
    return sorted(video_names)

def main():
    print("🧪 SAM2 PIPELINE SINGLE VIDEO TEST")
    print("=" * 60)
    
    # List available videos
    available_videos = list_available_videos()
    
    if not available_videos:
        print("❌ No videos found!")
        return
    
    # Use the first video with most annotations (basler_recording_20250710_092901)
    test_video = "basler_recording_20250710_092901"
    
    if test_video in available_videos:
        print(f"\n🎯 Testing with: {test_video}")
        test_single_video(test_video)
    else:
        print(f"\n🎯 Testing with: {available_videos[0]}")
        test_single_video(available_videos[0])

if __name__ == "__main__":
    main()
