from ultralytics import YOLO
import os
import torch

def train_yolo_segmentation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset')
    
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val

nc: 2
names: ['grove', 'wrod']"""
    
    yaml_path = os.path.join(script_dir, 'temp_config.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # model = YOLO('yolo12s.pt')  # only bbox
    model = YOLO('yolo11s-seg.pt')

    results = model.train(
        data=yaml_path,
        epochs=28,
        imgsz=1280,
        batch=16,
        device=device,
        project=os.path.join(script_dir, 'runs/segment'),
        name='weld_seg_1103-',
        save=True,
        save_period=4,
        patience=20,
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_yolo_segmentation()
