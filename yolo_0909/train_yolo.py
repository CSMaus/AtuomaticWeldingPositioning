from ultralytics import YOLO
import os
import torch

def train_yolo_segmentation():
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset')
    
    # Load pretrained model
    model = YOLO('yolo11s-seg.pt')
    
    # Train the model
    results = model.train(
        data={
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['grove_n_wrod']
        },
        epochs=20,
        imgsz=640,
        batch=8,
        device=device,
        project=os.path.join(script_dir, 'runs/segment'),
        name='weld_seg_0909',
        save=True,
        save_period=4,
        patience=20,
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_yolo_segmentation()
