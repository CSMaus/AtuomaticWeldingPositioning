import cv2
import numpy as np
from pypylon import pylon
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_camera_only():
    """Test just camera grabbing without any NN processing"""
    print("=== Testing Camera Only (No NN) ===")
    
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BayerBG8")
    camera.MaxNumBuffer = 3
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    frame_count = 0
    start_memory = get_memory_usage()
    total_time = 0
    
    try:
        while frame_count < 500:  # Test 500 frames
            start_time = time.time()
            
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                
                # Just display, no NN processing
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Camera Only Test', display_frame)
                
                frame_count += 1
                total_time += time.time() - start_time
                
                if frame_count % 100 == 0:
                    current_memory = get_memory_usage()
                    avg_time = total_time / frame_count
                    fps = 1.0 / avg_time
                    print(f"Frame {frame_count}: FPS: {fps:.1f}, Memory: {current_memory:.1f}MB (+{current_memory-start_memory:.1f}MB)")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        
        final_memory = get_memory_usage()
        print(f"Final: Memory increased by {final_memory-start_memory:.1f}MB over {frame_count} frames")

def test_with_nn():
    """Test with NN processing to see if that's the leak source"""
    print("\n=== Testing Camera + NN ===")
    
    from ultralytics import YOLO
    from helpers import get_masks_points_distance, draw_masks_points_distance
    
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BayerBG8")
    camera.MaxNumBuffer = 3
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    model = YOLO("yolo11s_labeling3/weights/best.pt")
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    
    frame_count = 0
    start_memory = get_memory_usage()
    total_time = 0
    
    try:
        while frame_count < 200:  # Fewer frames since NN is slower
            start_time = time.time()
            
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                
                # NN processing
                prediction = get_masks_points_distance(frame, 4.03, model, 0.0, resize_factor=0.5)
                labeled_frame = draw_masks_points_distance(frame, prediction, 0.0,
                                                         is_draw_masks=True,
                                                         is_draw_distance=True,
                                                         is_draw_groove_masks=True,
                                                         alpha=0.4)
                
                display_frame = cv2.cvtColor(labeled_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Camera + NN Test', display_frame)
                
                frame_count += 1
                total_time += time.time() - start_time
                
                if frame_count % 50 == 0:
                    current_memory = get_memory_usage()
                    avg_time = total_time / frame_count
                    fps = 1.0 / avg_time
                    print(f"Frame {frame_count}: FPS: {fps:.1f}, Memory: {current_memory:.1f}MB (+{current_memory-start_memory:.1f}MB)")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        
        final_memory = get_memory_usage()
        print(f"Final: Memory increased by {final_memory-start_memory:.1f}MB over {frame_count} frames")

if __name__ == "__main__":
    print("Memory leak diagnostic test")
    print("This will test camera only, then camera + NN to find the leak source")
    print("Press 'q' to skip to next test\n")
    
    test_camera_only()
    input("Press Enter to continue to NN test...")
    test_with_nn()
