import time
from pathlib import Path
import os
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLineEdit
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import get_masks_points_distance, get_masks_points_distance45, draw_masks_points_distance, write_json_file


path = os.path.join(Path.cwd().parents[2], "data/")

folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
videos_path = os.path.join(path, folders[0])
print(videos_path)

videos = [f for f in os.listdir(videos_path) if f.endswith(".mp4") and os.path.isfile(os.path.join(videos_path, f))]
curr_video_path = os.path.join(videos_path, videos[0])



class VideoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.videos = videos
        self.videos_path = videos_path
        self.video_path = os.path.join(self.videos_path, self.videos[0])
        self.cap = None
        self.num_iter = 0
        self.dt = 0
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Video Prediction GUI')

        layout = QVBoxLayout()

        cam_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["(new) W-Rod, Groove", "(old) YOLOv11-rotated45 - best", "(old) YOLOv11 - without rotation"])
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        self.load_selected_model()

        self.video_dropdown = QComboBox()
        self.video_dropdown.addItems(self.videos)
        self.video_dropdown.currentIndexChanged.connect(self.change_video)
        cam_layout.addWidget(QLabel("Video:"))
        cam_layout.addWidget(self.video_dropdown)

        cam_layout.addWidget(QLabel("Model:"))
        cam_layout.addWidget(self.model_selector)
        layout.addLayout(cam_layout)

        checkbox_layout = QHBoxLayout()
        self.mask_checkbox = QCheckBox("Draw Detections")
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

        self.start_btn = QPushButton("Start Video")
        self.start_btn.clicked.connect(self.toggle_video)
        layout.addWidget(self.start_btn)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        record_layout = QHBoxLayout()
        self.json_name_input = QLineEdit("predictions")
        record_layout.addWidget(QLabel("JSON Filename:"))
        record_layout.addWidget(self.json_name_input)

        self.record_checkbox = QCheckBox("Record JSON")
        record_layout.addWidget(self.record_checkbox)

        self.interval_input = QLineEdit("500")
        record_layout.addWidget(QLabel("Save every (ms):"))
        record_layout.addWidget(self.interval_input)

        layout.addLayout(record_layout)

        self.json_timer = QTimer()
        self.json_timer.timeout.connect(self.save_json_periodically)
        self.json_timer.start(500)
        self.interval_input.editingFinished.connect(self.update_json_timer_interval)

        self.latest_prediction = None
        self.setLayout(layout)

    def update_alpha_label(self):
        alpha = self.alpha_slider.value() / 100
        self.alpha_label.setText(f"{alpha:.2f}")

    def change_video(self):
        selected = self.video_dropdown.currentText()
        self.video_path = os.path.join(self.videos_path, selected)
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        print("Frame size:", int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def update_json_timer_interval(self):
        try:
            interval = int(self.interval_input.text().strip())
            self.json_timer.setInterval(interval if interval > 0 else 500)
        except ValueError:
            self.json_timer.setInterval(500)

    def toggle_video(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.start_btn.setText("Start Video")
            if self.num_iter != 0:
                print(f"Average time per frame with Resize Factor={self.resize_input.text()}: ", self.dt/self.num_iter)
                self.dt = 0
                self.num_iter = 0
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(30)
            self.start_btn.setText("Stop Video")
            if self.num_iter !=0:
                print(f"Average time per frame with Resize Factor={self.resize_input.text()}: ", self.dt/self.num_iter)
                self.dt = 0
                self.num_iter = 0

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        if model_name == "(old) YOLOv11-rotated45 - best":
            self.current_model = YOLO("electrode_groove_seg45/weights/best.pt")
        elif model_name == "(new) W-Rod, Groove":
            self.current_model = YOLO("yolo11s_labeling3/weights/best.pt")
        else:
            self.current_model = YOLO("electrode_groove_seg/weights/best.pt")
        self.current_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        try:
            angle = float(self.angle_input.text().strip())
        except ValueError:
            angle = 0.0

        try:
            width = float(self.width_input.text().strip())
        except ValueError:
            width = 0.0

        try:
            resize_factor = float(self.resize_input.text().strip())
        except ValueError:
            resize_factor = 1.0

        model_func = get_masks_points_distance45 if self.model_selector.currentText() == "(old) YOLOv11-rotated45 - best" else get_masks_points_distance
        st = time.time()
        prediction = model_func(frame, width, self.current_model, angle, resize_factor=resize_factor)

        # print("Frame took: ", en-st)
        labeled_frame = draw_masks_points_distance(frame, prediction, angle,
                                                   is_draw_masks=self.mask_checkbox.isChecked(),
                                                   is_draw_distance=self.distance_checkbox.isChecked(),
                                                   is_draw_groove_masks=self.grMask_checkbox.isChecked(),
                                                   alpha=self.alpha_slider.value() / 100)


        rgb_image = cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height()))
        if self.record_checkbox.isChecked():
            self.latest_prediction = prediction

        en = time.time()
        self.dt += en - st
        self.num_iter += 1

    def save_json_periodically(self):
        if self.record_checkbox.isChecked() and self.latest_prediction is not None:
            filename = self.json_name_input.text().strip() or "predictions"
            write_json_file(self.latest_prediction, filename)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoGUI()
    window.resize(800, 800)
    window.show()
    sys.exit(app.exec())
