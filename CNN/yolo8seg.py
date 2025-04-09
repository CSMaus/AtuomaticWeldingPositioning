from ultralytics import YOLO
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # this is bcs too many people works on this computer

if __name__ == '__main__':
    model = YOLO('yolo11n-seg.pt')
    # model = YOLO('runs/segment/electrode_groove_seg3/weights/last.pt')

    model.train(
        data='weld_data.yaml',
        imgsz=640,
        epochs=50,
        batch=8,
        name='electrode_groove_seg45',
        task='segment'
    )
    '''

    model = YOLO('yolov8n-seg.pt')

    results = model.train(
        data='weld_data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        name='electrode_groove_seg'
    )'''
