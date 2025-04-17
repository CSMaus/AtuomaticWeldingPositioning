# here is pyqt designed app to select the YOLO model and other params

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLineEdit
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import get_masks_points_distance, get_masks_points_distance45, draw_masks_points_distance

class CameraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Camera Prediction GUI')

        # self.model45 = YOLO("electrode_groove_seg45/weights/best.pt")
        # self.model0 = YOLO("electrode_groove_seg/weights/best.pt")
        # self.model0.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        # self.model45.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        self.cap = None

        layout = QVBoxLayout()

        cam_layout = QHBoxLayout()
        self.camera_dropdown = QComboBox()
        self.populate_cameras()
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

        self.setLayout(layout)

    def populate_cameras(self):
        self.camera_dropdown.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                self.camera_dropdown.addItem(f"Camera {i}", i)
            cap.release()

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
            self.start_btn.setText("Start")
        else:
            cam_idx = self.camera_dropdown.currentData()
            self.cap = cv2.VideoCapture(cam_idx)
            self.timer.start(30)
            self.start_btn.setText("Stop")

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        if model_name == "(Best) YOLOv11-rotated45":
            self.current_model = YOLO("electrode_groove_seg45/weights/best.pt")
        else:
            self.current_model = YOLO("electrode_groove_seg/weights/best.pt")
        self.current_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        angle = float(self.angle_input.text())
        width = float(self.width_input.text())

        model_func = get_masks_points_distance45 if self.model_selector.currentText() == "(Best) YOLOv11-rotated45" else get_masks_points_distance
        prediction = model_func(frame, width, self.current_model, angle)


        labeled_frame = draw_masks_points_distance(frame, prediction,
                                                   is_draw_masks=self.mask_checkbox.isChecked(),
                                                   is_draw_distance=self.distance_checkbox.isChecked())

        rgb_image = cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format).scaled(self.video_label.width(), self.video_label.height()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraGUI()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())



