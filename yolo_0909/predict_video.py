import cv2
import os
from ultralytics import YOLO

def predict_video():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    videos_dir = os.path.join(project_dir, "data", "basler_recordings")
    
    # List videos
    videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print("No videos found!")
        return
    
    print("Available videos:")
    for i, video in enumerate(videos):
        print(f"{i}: {video}")
    
    # Select video
    idx = int(input("Enter video index: "))
    video_path = os.path.join(videos_dir, videos[idx])
    
    # Load model (use best trained model)
    model_path = os.path.join(script_dir, "runs", "segment", "weld_seg_0909", "weights", "best.pt")
    if not os.path.exists(model_path):
        print("Trained model not found! Train first.")
        return
    
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            results = model(frame)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            cv2.imshow('YOLO Prediction', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' '):  # Space
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_video()
