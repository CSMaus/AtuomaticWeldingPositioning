import cv2
import os
from ultralytics import YOLO
import time

def predict_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    videos_dir = os.path.join(project_dir, "data", "basler_recordings")
    
    videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print("No videos found!")
        return
    
    print("Available videos:")
    for i, video in enumerate(videos):
        print(f"{i}: {video}")
    
    idx = int(input("Enter video index: "))
    video_path = os.path.join(videos_dir, videos[idx])
    
    model_path = os.path.join(script_dir, "runs", "segment", "weld_seg_09103", "weights", "best.pt")
    if not os.path.exists(model_path):
        print("Trained model not found! Train first.")
        return
    
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    paused = False
    fps_list = []
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            results = model(frame, verbose=False)
            inference_time = time.time() - start_time
            
            fps = 1.0 / inference_time
            fps_list.append(fps)
            
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            height, width = annotated_frame.shape[:2]
            new_width = int(width * 0.6)
            new_height = int(height * 0.6)
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
            
            cv2.imshow('YOLO Real-time Prediction', resized_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    predict_video()
