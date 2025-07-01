import time

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLineEdit
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import get_masks_points_distance, get_masks_points_distance45, draw_masks_points_distance, write_json_file
import sys
# pypylon for Basler camera
from pypylon import pylon
from time import time


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage, np.ndarray)

    def __init__(self):
        super().__init__()
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.PixelFormat.SetValue("BayerBG8")
        self.camera.MaxNumBuffer = 3
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.running = True
        self.target_fps = 10  # Limit to 10 FPS to prevent GUI freezing
        self.frame_interval = 1.0 / self.target_fps

    def run(self):
        last_frame_time = 0
        while self.running:
            current_time = time()
            
            # Frame rate limiting - only process if enough time has passed
            if current_time - last_frame_time < self.frame_interval:
                self.msleep(5)  # Sleep 5ms to prevent busy waiting
                continue
                
            grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)  # Shorter timeout
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                h, w, _ = frame.shape
                qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                
                # Emit both QImage for display and numpy array for NN processing
                self.frame_ready.emit(qimg, frame.copy())
                last_frame_time = current_time
                
                print(f"Frame emitted at {current_time:.3f}")

    def stop(self):
        self.running = False
        if self.camera is not None:
            self.camera.StopGrabbing()
            self.camera.Close()
        self.wait()


class CameraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_thread = None
        self.num_iter = 0
        self.dt = 0
        self.processing_frame = False  # Flag to prevent processing overlap

    def init_ui(self):
        self.setWindowTitle('Camera Prediction GUI - Anti-Freeze Version')
        
        layout = QVBoxLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["(new) W-Rod, Groove", "(old) YOLOv11-rotated45 - best", "(old) YOLOv11 - without rotation"])
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        self.load_selected_model()
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)

        # Checkboxes
        checkbox_layout = QHBoxLayout()
        self.mask_checkbox = QCheckBox("Draw Masks")
        self.mask_checkbox.setChecked(True)
        self.distance_checkbox = QCheckBox("Draw Distance")
        self.distance_checkbox.setChecked(True)
        self.grMask_checkbox = QCheckBox("Draw Groove Masks")
        self.grMask_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.mask_checkbox)
        checkbox_layout.addWidget(self.distance_checkbox)
        checkbox_layout.addWidget(self.grMask_checkbox)
        layout.addLayout(checkbox_layout)

        # Parameters
        input_layout = QHBoxLayout()
        self.angle_input = QLineEdit("0")
        self.width_input = QLineEdit("4.03")
        self.resize_input = QLineEdit("0.5")
        input_layout.addWidget(QLabel("Angle:"))
        input_layout.addWidget(self.angle_input)
        input_layout.addWidget(QLabel("Width:"))
        input_layout.addWidget(self.width_input)
        input_layout.addWidget(QLabel("Resize:"))
        input_layout.addWidget(self.resize_input)
        layout.addLayout(input_layout)

        # Video display
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.video_label)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.start_btn)
        
        self.status_label = QLabel("Ready")
        button_layout.addWidget(self.status_label)
        layout.addLayout(button_layout)

        # JSON recording
        record_layout = QHBoxLayout()
        self.json_name_input = QLineEdit("predictions")
        self.record_checkbox = QCheckBox("Record JSON")
        record_layout.addWidget(QLabel("JSON File:"))
        record_layout.addWidget(self.json_name_input)
        record_layout.addWidget(self.record_checkbox)
        layout.addLayout(record_layout)

        self.setLayout(layout)

        # JSON timer
        self.json_timer = QTimer()
        self.json_timer.timeout.connect(self.save_json_periodically)
        self.json_timer.start(1000)  # Save every 1 second instead of 500ms
        self.latest_prediction = None

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        self.status_label.setText("Loading model...")
        QApplication.processEvents()  # Force GUI update
        
        if model_name == "(old) YOLOv11-rotated45 - best":
            self.current_model = YOLO("electrode_groove_seg45/weights/best.pt")
        elif model_name == "(new) W-Rod, Groove":
            self.current_model = YOLO("yolo11s_labeling3/weights/best.pt")
        else:
            self.current_model = YOLO("electrode_groove_seg/weights/best.pt")
            
        # Warm up model
        self.current_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        self.status_label.setText("Model loaded")

    def toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None
            self.start_btn.setText("Start Camera")
            self.status_label.setText("Camera stopped")
        else:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_image)
            self.camera_thread.start()
            self.start_btn.setText("Stop Camera")
            self.status_label.setText("Camera running")

    def update_image(self, qimg, frame):
        # Skip processing if already processing a frame (prevent backlog)
        if self.processing_frame:
            return
            
        self.processing_frame = True
        
        try:
            st = time.time()
            
            # Get parameters
            try:
                angle = float(self.angle_input.text().strip())
                width = float(self.width_input.text().strip())
                resize_factor = float(self.resize_input.text().strip())
            except ValueError:
                angle, width, resize_factor = 0.0, 4.03, 0.5

            # Choose model function
            model_func = get_masks_points_distance45 if self.model_selector.currentText() == "(old) YOLOv11-rotated45 - best" else get_masks_points_distance

            # Process with YOLO
            prediction = model_func(frame, width, self.current_model, angle, resize_factor=resize_factor)
            
            # Draw results
            labeled_frame = draw_masks_points_distance(frame, prediction, angle,
                                                       is_draw_masks=self.mask_checkbox.isChecked(),
                                                       is_draw_distance=self.distance_checkbox.isChecked(),
                                                       is_draw_groove_masks=self.grMask_checkbox.isChecked(),
                                                       alpha=0.4)

            # Convert to QImage for display
            h, w, ch = labeled_frame.shape
            bytes_per_line = ch * w
            processed_qimg = QImage(labeled_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # Display
            pixmap = QPixmap.fromImage(processed_qimg)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            
            # Save prediction
            if self.record_checkbox.isChecked():
                self.latest_prediction = prediction
            
            # Update status
            en = time.time()
            processing_time = en - st
            self.status_label.setText(f"Processing: {processing_time:.3f}s")
            
        except Exception as e:
            print(f"Error in update_image: {e}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.processing_frame = False

    def save_json_periodically(self):
        if self.record_checkbox.isChecked() and self.latest_prediction is not None:
            try:
                name_text = self.json_name_input.text().strip()
                json_file_name = name_text if name_text else "predictions"
                write_json_file(self.latest_prediction, json_file_name)
            except Exception as e:
                print(f"Error saving JSON: {e}")

    def closeEvent(self, event):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraGUI()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec())
