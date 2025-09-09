from ultralytics import YOLO
import os
import torch

def train_yolo_segmentation():
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, 'weld_seg.yaml')
    
    # Load pretrained model
    model = YOLO('yolo11n-seg.pt')
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project=os.path.join(script_dir, 'runs/segment'),
        name='weld_seg_0909',
        save=True,
        save_period=10,
        patience=20,
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_yolo_segmentation()
