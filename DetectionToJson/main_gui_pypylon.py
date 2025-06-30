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
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.PixelFormat.SetValue("BayerBG8")
        self.camera.MaxNumBuffer = 3
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.running = True

    def run(self):
        while self.running:
            st = time()
            grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                h, w, _ = frame.shape
                qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
            ed = time()
            print("Image grab took:", ed - st)

    def stop(self):
        self.running = False
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

    def init_ui(self):
        self.setWindowTitle('Camera Prediction GUI')
        self.cap = None

        layout = QVBoxLayout()

        cam_layout = QHBoxLayout()
        self.camera_dropdown = QComboBox()
        self.populate_cameras()
        self.camera_dropdown.currentIndexChanged.connect(self.change_camera)
        cam_layout.addWidget(QLabel("Camera:"))
        cam_layout.addWidget(self.camera_dropdown)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["(new) W-Rod, Groove", "(old) YOLOv11-rotated45 - best", "(old) YOLOv11 - without rotation"])
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        self.load_selected_model()
        cam_layout.addWidget(QLabel("Model:"))
        cam_layout.addWidget(self.model_selector)

        layout.addLayout(cam_layout)

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

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Groove Mask Opacity:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setFixedHeight(20)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(40)
        self.alpha_slider.setSingleStep(1)
        self.alpha_label = QLabel("0.40")
        self.alpha_label.setFixedHeight(20)
        self.alpha_slider.valueChanged.connect(self.update_alpha_label)

        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_label)
        layout.addLayout(alpha_layout)

        input_layout = QHBoxLayout()
        self.angle_input = QLineEdit("0")
        self.width_input = QLineEdit("4.03")
        input_layout.addWidget(QLabel("Camera Rotation (deg):"))
        input_layout.addWidget(self.angle_input)
        input_layout.addWidget(QLabel("Electrode Width (mm):"))
        input_layout.addWidget(self.width_input)

        input_layout.addWidget(QLabel("Resize Factor:"))
        self.resize_input = QLineEdit("0.5")
        input_layout.addWidget(self.resize_input)

        layout.addLayout(input_layout)

        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_camera)
        layout.addWidget(self.start_btn)

        # Remove timer setup - we'll use thread instead
        record_layout = QHBoxLayout()
        self.json_name_input = QLineEdit("predictions")
        record_layout.addWidget(QLabel("JSON Filename:"))
        record_layout.addWidget(self.json_name_input)
        self.record_checkbox = QCheckBox("Record JSON")
        record_layout.addWidget(self.record_checkbox)
        layout.addLayout(record_layout)

        self.json_timer = QTimer()
        self.json_timer.timeout.connect(self.save_json_periodically)
        self.json_timer.start(500)
        self.latest_prediction = None
        self.interval_input = QLineEdit("500")
        record_layout.addWidget(QLabel("Save every (ms):"))
        record_layout.addWidget(self.interval_input)


        self.interval_input.editingFinished.connect(self.update_json_timer_interval)
        self.setLayout(layout)




    def update_alpha_label(self):
        alpha = self.alpha_slider.value() / 100
        self.alpha_label.setText(f"{alpha:.2f}")

    def change_camera(self):
        # This method is no longer needed as camera setup is handled in thread
        pass

    def update_json_timer_interval(self):
        try:
            interval_text = self.interval_input.text().strip()
            interval = int(interval_text) if interval_text else 500
            if interval > 0:
                self.json_timer.setInterval(interval)
        except ValueError:
            self.json_timer.setInterval(500)

    def populate_cameras(self):
        self.camera_dropdown.clear()
        # Basler
        self.camera_dropdown.addItem("Basler GigE", "basler")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                self.camera_dropdown.addItem(f"Camera {i}", i)
            cap.release()

    def toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None
            self.start_btn.setText("Start")
            if self.num_iter != 0:
                print(f"Average time per frame with Resize Factor={self.resize_input.text()}: ", self.dt / self.num_iter)
                self.dt = 0
                self.num_iter = 0
        else:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_image)
            self.camera_thread.start()
            self.start_btn.setText("Stop")

    def release_camera(self):
        # This method is no longer needed as cleanup is handled in thread
        pass

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        if model_name == "(old) YOLOv11-rotated45 - best":
            self.current_model = YOLO("electrode_groove_seg45/weights/best.pt")
        elif model_name == "(new) W-Rod, Groove":
            self.current_model = YOLO("yolo11s_labeling3/weights/best.pt")
        else:
            self.current_model = YOLO("electrode_groove_seg/weights/best.pt")
        self.current_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)


    def update_image(self, qimg):  # MUST BE update_image like working code!
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # For timing measurement
        self.num_iter += 1

    def save_json_periodically(self):
        if self.record_checkbox.isChecked() and self.latest_prediction is not None:
            name_text = self.json_name_input.text().strip()
            json_file_name = name_text if name_text else "predictions"
            write_json_file(self.latest_prediction, json_file_name)

    def closeEvent(self, event):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraGUI()
    window.resize(800, 800)
    window.show()
    sys.exit(app.exec())