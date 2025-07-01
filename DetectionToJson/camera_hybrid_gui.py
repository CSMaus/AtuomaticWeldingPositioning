"""
Hybrid approach: OpenCV display (no freezing) + simple PyQt6 control panel
"""
import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO
from helpers import get_masks_points_distance, get_masks_points_distance45, draw_masks_points_distance, write_json_file
import time
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLineEdit, QLabel
from PyQt6.QtCore import QTimer
import threading

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_running = False
        self.camera_thread = None
        
        # Shared parameters (thread-safe)
        self.params = {
            'angle': 0.0,
            'width': 4.03,
            'resize_factor': 0.5,
            'draw_masks': True,
            'draw_distance': True,
            'draw_groove_masks': True,
            'record_json': False,
            'json_filename': 'predictions',
            'model_type': 0
        }
        
        self.load_model()
        
    def init_ui(self):
        self.setWindowTitle('Camera Control Panel')
        layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["(new) W-Rod, Groove", "(old) YOLOv11-rotated45 - best", "(old) YOLOv11 - without rotation"])
        self.model_selector.currentIndexChanged.connect(self.change_model)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)
        
        # Parameters
        param_layout = QHBoxLayout()
        self.angle_input = QLineEdit("0")
        self.width_input = QLineEdit("4.03")
        self.resize_input = QLineEdit("0.5")
        param_layout.addWidget(QLabel("Angle:"))
        param_layout.addWidget(self.angle_input)
        param_layout.addWidget(QLabel("Width:"))
        param_layout.addWidget(self.width_input)
        param_layout.addWidget(QLabel("Resize:"))
        param_layout.addWidget(self.resize_input)
        layout.addLayout(param_layout)
        
        # Checkboxes
        check_layout = QHBoxLayout()
        self.mask_check = QCheckBox("Draw Masks")
        self.mask_check.setChecked(True)
        self.distance_check = QCheckBox("Draw Distance")
        self.distance_check.setChecked(True)
        self.groove_check = QCheckBox("Draw Groove Masks")
        self.groove_check.setChecked(True)
        check_layout.addWidget(self.mask_check)
        check_layout.addWidget(self.distance_check)
        check_layout.addWidget(self.groove_check)
        layout.addLayout(check_layout)
        
        # JSON recording
        json_layout = QHBoxLayout()
        self.json_input = QLineEdit("predictions")
        self.record_check = QCheckBox("Record JSON")
        json_layout.addWidget(QLabel("JSON File:"))
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(self.record_check)
        layout.addLayout(json_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        self.update_btn = QPushButton("Update Parameters")
        self.update_btn.clicked.connect(self.update_parameters)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.update_btn)
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Auto-update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_parameters)
        self.timer.start(100)  # Update parameters every 100ms
        
    def load_model(self):
        self.status_label.setText("Loading model...")
        QApplication.processEvents()
        
        model_names = ["yolo11s_labeling3/weights/best.pt", 
                      "electrode_groove_seg45/weights/best.pt",
                      "electrode_groove_seg/weights/best.pt"]
        
        self.model = YOLO(model_names[self.params['model_type']])
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        self.status_label.setText("Model loaded")
        
    def change_model(self):
        self.params['model_type'] = self.model_selector.currentIndex()
        self.load_model()
        
    def update_parameters(self):
        try:
            self.params['angle'] = float(self.angle_input.text())
            self.params['width'] = float(self.width_input.text())
            self.params['resize_factor'] = float(self.resize_input.text())
            self.params['draw_masks'] = self.mask_check.isChecked()
            self.params['draw_distance'] = self.distance_check.isChecked()
            self.params['draw_groove_masks'] = self.groove_check.isChecked()
            self.params['record_json'] = self.record_check.isChecked()
            self.params['json_filename'] = self.json_input.text()
        except ValueError:
            pass
            
    def toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.start_btn.setText("Start Camera")
            self.status_label.setText("Stopping camera...")
        else:
            self.camera_running = True
            self.start_btn.setText("Stop Camera")
            self.status_label.setText("Starting camera...")
            self.camera_thread = threading.Thread(target=self.run_camera, daemon=True)
            self.camera_thread.start()
            
    def run_camera(self):
        """Run camera in separate thread with OpenCV display"""
        # Initialize camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        camera.PixelFormat.SetValue("BayerBG8")
        camera.MaxNumBuffer = 3
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        frame_count = 0
        total_time = 0
        last_json_save = time.time()
        latest_prediction = None
        
        try:
            while self.camera_running:
                start_time = time.time()
                
                grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
                if grab_result and grab_result.GrabSucceeded():
                    img = grab_result.Array.copy()
                    grab_result.Release()
                    frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                    
                    # Get current parameters (thread-safe read)
                    params = self.params.copy()
                    
                    # Process with YOLO
                    if params['model_type'] == 1:  # rotated45 model
                        prediction = get_masks_points_distance45(frame, params['width'], self.model, 
                                                               params['angle'], resize_factor=params['resize_factor'])
                    else:
                        prediction = get_masks_points_distance(frame, params['width'], self.model, 
                                                             params['angle'], resize_factor=params['resize_factor'])
                    
                    # Draw results
                    labeled_frame = draw_masks_points_distance(frame, prediction, params['angle'],
                                                             is_draw_masks=params['draw_masks'],
                                                             is_draw_distance=params['draw_distance'],
                                                             is_draw_groove_masks=params['draw_groove_masks'],
                                                             alpha=0.4)
                    
                    # Convert for OpenCV display
                    display_frame = cv2.cvtColor(labeled_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Camera + NN (OpenCV)', display_frame)
                    
                    # JSON recording
                    if params['record_json'] and time.time() - last_json_save > 1.0:
                        try:
                            write_json_file(prediction, params['json_filename'])
                            last_json_save = time.time()
                        except Exception as e:
                            print(f"JSON save error: {e}")
                    
                    # Timing
                    frame_time = time.time() - start_time
                    total_time += frame_time
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        avg_fps = frame_count / total_time if total_time > 0 else 0
                        self.status_label.setText(f"Running - FPS: {avg_fps:.1f}")
                
                # Handle OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.camera_running = False
                    break
                    
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            camera.StopGrabbing()
            camera.Close()
            cv2.destroyAllWindows()
            self.status_label.setText("Camera stopped")

def main():
    app = QApplication(sys.argv)
    
    control_panel = ControlPanel()
    control_panel.show()
    
    print("Control Panel opened. Use it to start camera.")
    print("Camera will open in separate OpenCV window.")
    print("Press 'q' in camera window or use control panel to stop.")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
