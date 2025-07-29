from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam2_b.pt")

# Use prompts to segment specific objects in images or videos.
from ultralytics import SAM
# Load a model
model = SAM("sam2.1_b.pt")
# Display model information (optional)
model.info()
# Run inference with bboxes prompt
results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])
# Run inference with single point
results = model(points=[900, 370], labels=[1])
# Run inference with multiple points
results = model(points=[[400, 370], [900, 370]], labels=[1, 1])
# Run inference with multiple points prompt per object
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])
# Run inference with negative points prompt
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])



# Segment Video and Track objects
from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

# Run inference with single point
results = predictor(source="test.mp4", points=[920, 470], labels=[1])

# Run inference with multiple points
results = predictor(source="test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

# Run inference with multiple points prompt per object
results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

# Run inference with negative points prompt
results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])



