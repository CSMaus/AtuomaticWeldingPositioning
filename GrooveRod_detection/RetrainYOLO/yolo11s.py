from ultralytics import YOLO
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # this is bcs too many people works on this computer

if __name__ == '__main__':
    model = YOLO('yolo11n-seg.pt')
    # model = YOLO('runs/segment/electrode_groove_seg3/weights/last.pt')

    model.train(
        data='data.yaml',
        imgsz=640,
        epochs=50,
        batch=8,
        name='yolo11s_labeling',
        task='segment'
    )