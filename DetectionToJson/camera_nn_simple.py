import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO
from helpers import get_masks_points_distance, get_masks_points_distance45, draw_masks_points_distance
import time
import gc  # Add garbage collection

def main():
    # Initialize camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BayerBG8")
    camera.MaxNumBuffer = 3
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO("yolo11s_labeling3/weights/best.pt")  # Change to your preferred model
    # Warm up the model
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("Model loaded!")
    
    # Parameters
    angle = 0.0  # Camera rotation in degrees
    width = 4.03  # Electrode width in mm
    resize_factor = 0.5  # Resize factor for processing
    
    # Timing variables
    frame_count = 0
    total_time = 0
    win_name = "Basler camera + NN"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)
    
    print("Starting camera stream... Press 'q' to quit")
    print("Press 's' to save current prediction to JSON")
    
    try:
        while True:
            start_time = time.time()
            
            # Grab frame from camera
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                # Convert camera image
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                
                # Process with YOLO
                prediction = get_masks_points_distance(frame, width, model, angle, resize_factor=resize_factor)
                
                # Draw results on frame
                labeled_frame = draw_masks_points_distance(frame, prediction, angle,
                                                         is_draw_masks=True,
                                                         is_draw_distance=True,
                                                         is_draw_groove_masks=True,
                                                         alpha=0.4)
                
                # Convert RGB to BGR for OpenCV display
                display_frame = cv2.cvtColor(labeled_frame, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow(win_name, display_frame)
                
                # Force garbage collection every 100 frames to prevent memory buildup
                if frame_count % 100 == 0:
                    gc.collect()
                
                # Timing
                end_time = time.time()
                frame_time = end_time - start_time
                total_time += frame_time
                frame_count += 1
                
                # Print timing info every 30 frames
                if frame_count % 30 == 0:
                    avg_time = total_time / frame_count
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"Frame {frame_count}: Avg time per frame: {avg_time:.3f}s, FPS: {fps:.1f}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save prediction to JSON
                    from helpers import write_json_file
                    write_json_file(prediction, "simple_camera_prediction")
                    print("Prediction saved to JSON!")
            
            else:
                print("Failed to grab frame")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_time = total_time / frame_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\nFinal stats:")
            print(f"Total frames: {frame_count}")
            print(f"Average time per frame: {avg_time:.3f}s")
            print(f"Average FPS: {fps:.1f}")

if __name__ == "__main__":
    main()
