from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLineEdit
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import get_masks_points_distance, get_masks_points_distance45, draw_masks_points_distance, write_json_file
import sys
# pypylon for Basler camera
from pypylon import pylon

class CameraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.is_basler = False
        self.basler_cam = None

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
        self.model_selector.addItems(["(Best) YOLOv11-rotated45", "YOLOv11 - without rotation"])
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
        checkbox_layout.addWidget(self.mask_checkbox)
        checkbox_layout.addWidget(self.distance_checkbox)
        layout.addLayout(checkbox_layout)
        input_layout = QHBoxLayout()
        self.angle_input = QLineEdit("0")
        self.width_input = QLineEdit("4.03")
        input_layout.addWidget(QLabel("Camera Rotation (deg):"))
        input_layout.addWidget(self.angle_input)
        input_layout.addWidget(QLabel("Electrode Width (mm):"))
        input_layout.addWidget(self.width_input)
        layout.addLayout(input_layout)
        self.video_label = QLabel()
        layout.addWidget(self.video_label)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_camera)
        layout.addWidget(self.start_btn)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
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

    def change_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.release_camera()
        cam_idx = self.camera_dropdown.currentData()
        if cam_idx == "basler":
            self.is_basler = True
            self.basler_cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.basler_cam.Open()
            self.basler_cam.StartGrabbing()
        else:
            self.is_basler = False
            self.cap = cv2.VideoCapture(cam_idx)
        self.timer.start(30)

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
        # Add Basler as option
        self.camera_dropdown.addItem("Basler GigE", "basler")
        # Add normal USB cameras
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                self.camera_dropdown.addItem(f"Camera {i}", i)
            cap.release()

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.release_camera()
            self.start_btn.setText("Start")
        else:
            self.change_camera()
            self.start_btn.setText("Stop")

    def release_camera(self):
        if self.is_basler and self.basler_cam is not None:
            self.basler_cam.StopGrabbing()
            self.basler_cam.Close()
            self.basler_cam = None
        if not self.is_basler and self.cap is not None:
            self.cap.release()
            self.cap = None

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        if model_name == "(Best) YOLOv11-rotated45":
            self.current_model = YOLO("electrode_groove_seg45/weights/best.pt")
        else:
            self.current_model = YOLO("electrode_groove_seg/weights/best.pt")
        self.current_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    def update_frame(self):
        if self.is_basler and self.basler_cam is not None and self.basler_cam.IsGrabbing():
            grab = self.basler_cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = grab.Array
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                grab.Release()
            else:
                return
        elif not self.is_basler and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return
        else:
            return

        try:
            angle = float(self.angle_input.text().strip())
        except ValueError:
            angle = 0.0
        try:
            width = float(self.width_input.text().strip())
        except ValueError:
            width = 0.0

        model_func = get_masks_points_distance45 if self.model_selector.currentText() == "(Best) YOLOv11-rotated45" else get_masks_points_distance
        prediction = model_func(frame, width, self.current_model, angle)
        labeled_frame = draw_masks_points_distance(frame, prediction, angle,
                                                   is_draw_masks=self.mask_checkbox.isChecked(),
                                                   is_draw_distance=self.distance_checkbox.isChecked())

        rgb_image = cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format).scaled(self.video_label.width(), self.video_label.height()))
        if self.record_checkbox.isChecked():
            self.latest_prediction = prediction

    def save_json_periodically(self):
        if self.record_checkbox.isChecked() and self.latest_prediction is not None:
            name_text = self.json_name_input.text().strip()
            json_file_name = name_text if name_text else "predictions"
            write_json_file(self.latest_prediction, json_file_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraGUI()
    window.resize(800, 800)
    window.show()
    sys.exit(app.exec())