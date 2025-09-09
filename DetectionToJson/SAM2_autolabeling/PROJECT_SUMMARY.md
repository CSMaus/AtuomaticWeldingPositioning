# SAM2 Video Auto-labeling Project Summary

## Initial Task
User wanted to create SAM2 video segmentation for welding groove detection with proper frame mapping and comprehensive annotation saving.

## Key Problems Identified
1. **Frame Index Mismatch**: Image frame numbers in filenames (frame_0000, frame_0001) were NOT the same as actual video frame indices
2. **Missing Annotations**: Existing SAM2 scripts only saved annotated videos, not individual annotation files
3. **Incomplete Predictions**: Scripts were only saving first prediction per frame, ignoring multiple detections
4. **No Resume Capability**: Processing would lose progress if interrupted

## Project Structure
```
/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/DetectionToJson/SAM2_autolabeling/
├── basler_recordings/                    # Original video files (.avi)
├── basler_recordings_frames/             # Extracted frame images (wrong indices)
├── annotations.xml                       # Original CVAT annotations
├── corrected_annotations/                # Fixed frame mappings and annotations
│   └── {video_name}/
│       ├── frames/                       # Corrected frame images
│       ├── {video_name}_corrected_annotations.json
│       └── {video_name}_corrected_annotations.xml
├── sam2_results/                         # SAM2 auto-annotation results
└── sam2_corrected_results/               # SAM2 results using corrected annotations
```

## Scripts Created

### 1. `frame_mapping_and_correction.py`
**Purpose**: Map image frame numbers to correct video frame indices
**Key Features**:
- Uses template matching with 0.98+ similarity threshold
- Finds exact correspondence between image files and video frames
- Creates corrected annotations with proper frame indices
- Copies images with correct frame naming

**Output**: 
- `corrected_annotations/{video_name}/` folders
- Corrected JSON and XML annotation files
- Frame images with correct indices

### 2. `sam2.py` (Main Auto-annotation Script)
**Purpose**: Process videos frame-by-frame using SAM2 with corrected annotations as prompts
**Key Features**:
- Loads corrected annotations as reference
- Extracts polygon center points as SAM2 prompts
- Processes ALL video frames (not just annotated ones)
- Saves ALL predictions per frame (not just first one)
- Resume capability - skips already processed frames
- Progress saving every 10 frames

**Input**: Corrected annotations from `corrected_annotations/`
**Output**: `sam2_results/{video_name}_annotations.json`

**JSON Structure**:
```json
{
  "frame": 42,
  "annotations": [
    {
      "result_index": 0,
      "mask_index": 0, 
      "polygon_points": [[x1,y1], [x2,y2], ...],
      "mask_area": 1234,
      "polygon_points_count": 15
    }
  ],
  "total_annotations": 3,
  "is_reference_frame": true
}
```

### 3. `vis.py` (Visualization Script)
**Purpose**: Visualize SAM2 results on video frames
**Key Features**:
- Loads results from `sam2_results/`
- Displays ALL polygons per frame with different colors
- Shows reference frames vs auto-generated frames
- Interactive controls (play/pause, next/prev, reference-only mode)

**Controls**:
- `q` - Quit
- `n`/`p` - Next/Previous frame
- `SPACE` - Play/Pause
- `r` - Show reference frames only
- `a` - Show all frames

### 4. `sam2_corrected_tracker.py` (Alternative Video Tracker)
**Purpose**: SAM2VideoPredictor-based approach (alternative to frame-by-frame)
**Features**: Uses SAM2's video tracking capabilities with corrected annotations

## Key Technical Solutions

### Frame Mapping Algorithm
```python
# Template matching with high threshold
result = cv2.matchTemplate(target_gray, frame_gray, cv2.TM_CCOEFF_NORMED)
_, max_val, _, _ = cv2.minMaxLoc(result)

if max_val > 0.98:  # High similarity threshold
    return frame_idx  # Found exact match
```

### Multiple Predictions Handling
```python
# Process ALL results and ALL masks
for result_idx, result in enumerate(results):
    if hasattr(result, 'masks') and result.masks is not None:
        for mask_idx, mask in enumerate(result.masks.data):
            # Save each mask as separate annotation
```

### Resume Capability
```python
# Check existing annotations and skip processed frames
existing_frames = {ann['frame'] for ann in existing_annotations}
frames_to_process = [f for f in range(total_frames) if f not in existing_frames]
```

## What We Achieved

1. **Correct Frame Mapping**: 
   - Mapped image frame numbers to actual video frame indices
   - Example: `frame_0000.png` → video frame 42, `frame_0001.png` → video frame 121

2. **Comprehensive Auto-annotation**:
   - SAM2 processes entire videos using corrected reference frames
   - Saves detailed JSON annotations for every frame
   - Captures ALL predictions per frame (not just first one)

3. **Resume-Safe Processing**:
   - Can interrupt and resume processing without losing progress
   - Incremental saving every 10 frames

4. **Multiple Output Formats**:
   - JSON annotations with detailed polygon/mask data
   - Visualization capabilities for result verification

## Current Status

### Working Scripts:
- ✅ `frame_mapping_and_correction.py` - Creates corrected annotations
- ✅ `sam2.py` - Main auto-annotation script (fixed to save ALL predictions)
- ✅ `vis.py` - Visualization script (fixed to display ALL polygons)

### Verified Functionality:
- Frame mapping with high accuracy (0.98+ similarity)
- SAM2 processing with multiple predictions per frame
- JSON annotation saving with resume capability
- Interactive visualization of results

## Usage Workflow

1. **Run Frame Mapping**:
   ```bash
   python frame_mapping_and_correction.py
   ```

2. **Run SAM2 Auto-annotation**:
   ```bash
   python sam2.py
   ```

3. **Visualize Results**:
   ```bash
   python vis.py
   ```

## Key Files for Resume

- **Main Scripts**: `sam2.py`, `vis.py`, `frame_mapping_and_correction.py`
- **Input Data**: `corrected_annotations/` folder
- **Output Data**: `sam2_results/` folder
- **Configuration**: Video paths, model settings in script headers

## Technical Notes

- **SAM2 Model**: Uses `sam2.1_l.pt` for high accuracy
- **Frame Processing**: Template matching for exact frame correspondence
- **Memory Management**: Frame-by-frame processing to avoid RAM issues
- **Error Handling**: Try-catch blocks with detailed error reporting
- **Progress Tracking**: Console output and incremental file saving

## Next Steps (if needed)
- Batch processing for multiple videos
- Performance optimization for large video datasets
- Integration with training pipelines
- Quality metrics and validation tools
