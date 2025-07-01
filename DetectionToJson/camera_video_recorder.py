import cv2
import numpy as np
from pypylon import pylon
import time
import os
from datetime import datetime

def main():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BayerBG8")
    camera.MaxNumBuffer = 3
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    print("Getting camera frame size...")
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        img = grab_result.Array.copy()
        grab_result.Release()
        frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        height, width, channels = frame.shape
        print(f"Camera resolution: {width}x{height}")
    else:
        print("Failed to get initial frame")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 'MP4V', 'MJPG', XVID
    fps = 20.0  # Target FPS
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"basler_recording_{timestamp}.avi"
    
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return
    
    print(f"Recording to: {video_filename}")
    print("Controls:")
    print("  SPACE - Start/Stop recording")
    print("  'q' - Quit")
    print("  's' - Save single frame as image")
    
    is_recording = False
    frame_count = 0
    recorded_frames = 0
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # recording indicator
                if is_recording:
                    cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    video_writer.write(display_frame)
                    recorded_frames += 1
                else:
                    cv2.circle(display_frame, (30, 30), 15, (128, 128, 128), -1)
                    cv2.putText(display_frame, "READY", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                # FPS info
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                info_text = f"Frame: {frame_count} | FPS: {current_fps:.1f}"
                if is_recording:
                    info_text += f" | Recorded: {recorded_frames}"
                
                cv2.putText(display_frame, info_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Basler Camera Recorder', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    is_recording = not is_recording
                    if is_recording:
                        print(f"Started recording at frame {frame_count}")
                    else:
                        print(f"Stopped recording. Recorded {recorded_frames} frames")
                elif key == ord('s'):
                    frame_filename = f"basler_frame_{timestamp}_{frame_count:06d}.jpg"
                    cv2.imwrite(frame_filename, display_frame)
                    print(f"Saved frame: {frame_filename}")
                
                if frame_count % 100 == 0:
                    status = "RECORDING" if is_recording else "READY"
                    print(f"Frame {frame_count} | FPS: {current_fps:.1f} | Status: {status}")
            
            else:
                print("Failed to grab frame")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        if is_recording:
            print(f"Finalizing recording... {recorded_frames} frames recorded")
        
        video_writer.release()
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\nRecording session complete:")
        print(f"  Video file: {video_filename}")
        print(f"  Total frames captured: {frame_count}")
        print(f"  Frames recorded to video: {recorded_frames}")
        print(f"  Session duration: {total_time:.1f} seconds")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        if recorded_frames > 0:
            file_size = os.path.getsize(video_filename) / (1024 * 1024)  # MB
            print(f"  Video file size: {file_size:.1f} MB")

if __name__ == "__main__":
    main()
