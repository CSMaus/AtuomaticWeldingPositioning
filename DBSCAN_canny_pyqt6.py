import os
import cv2
import numpy as np
import random
from sklearn.cluster import DBSCAN
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import sys
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

videos_path = "/Users/kseni/Downloads/kakao/Robot REC/"
# videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
# this_video_path = os.path.join(videos_path, os.listdir(videos_path)[10])
this_video_path = os.path.join(videos_path, "rb_test7.mp4")  # "rb_test7.mp4")  # "rb6.360mm & 30d.mp4"

cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_paused = False
frame_pos = 0
last_good_center = None
last_good_index = None

# ############################
cluster_history = []
MAX_HISTORY = 5
num_clusters_to_ex = 2

# here will be the code for the joint line detection

def get_cluster_centers(clusters):
    return [np.mean(c.reshape(-1, 2), axis=0) for c in clusters]

def find_consistent_cluster_index(cluster_centers, last_center, max_distance=40):
    if last_center is None or not cluster_centers:
        return 0
    best_index = None
    best_distance = float('inf')
    for i, curr in enumerate(cluster_centers):
        dist = np.linalg.norm(np.array(curr) - np.array(last_center))
        if dist < best_distance and dist < max_distance:
            best_distance = dist
            best_index = i
    return best_index if best_index is not None else 0

def extract_all_clusters(edges, eps, min_samples, leaf_size):
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) == 0:
        return []
    points = np.stack((x_indices, y_indices), axis=1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size).fit(points)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_pts = points[labels == label]
        clusters.append(cluster_pts.reshape(-1, 1, 2))
    return clusters

def ransac_polynomial_curve(points, degree=2, residual_threshold=5):
    if len(points) < degree + 1:
        return None, None
    points = points.reshape(-1, 2)
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = make_pipeline(PolynomialFeatures(degree),
                          RANSACRegressor(residual_threshold=residual_threshold,
                                          random_state=42))
    model.fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_range = model.predict(x_range)
    return x_range.astype(int), y_range.astype(int)

def find_lowest_point(x_fit, y_fit):
    x_fit = x_fit.flatten()
    y_fit = y_fit.flatten()
    idx = np.argmax(y_fit)  # bottom y is the derivative
    return int(x_fit[idx].item()), int(y_fit[idx].item())

previous_bottom = None  # define globally
def find_lowest_point_smooth(x_fit, y_fit, alpha=0.7):
    global previous_bottom
    x_fit = x_fit.flatten()
    y_fit = y_fit.flatten()
    idx = np.argmax(y_fit)
    new_bottom = np.array([x_fit[idx].item(), y_fit[idx].item()])
    if previous_bottom is None:
        previous_bottom = new_bottom
    else:
        previous_bottom = alpha * previous_bottom + (1 - alpha) * new_bottom
    return tuple(previous_bottom.astype(int))

def process_frame(frame, params):
    global last_good_center, last_good_index, cluster_history
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, params['d'], params['sigmaColor'], params['sigmaSpace'])
    edges = cv2.Canny(filtered, params['canny_min'], params['canny_max'])

    if num_clusters_to_ex < 2:
        # this one extracts only one part. So, there will be contours only for electrode or only for the part below it
        y_indices, x_indices = np.where(edges > 0)
        if len(x_indices) == 0:
            return frame
        points = np.stack((x_indices, y_indices), axis=1)
        clustering = DBSCAN(eps=15, min_samples=10).fit(points)
        labels = clustering.labels_
        if np.max(labels) >= 0:
            best_label = np.argmax(np.bincount(labels[labels >= 0]))
            selected = points[labels == best_label].reshape(-1, 1, 2)
            for pt in selected:
                cv2.circle(frame, tuple(pt[0]), 1, (0, 255, 0), -1)
    else:
        clusters = extract_all_clusters(edges, params['epsilon'], params['min_samples'], params['leaf_size'])
        centers = get_cluster_centers(clusters)
        idx = find_consistent_cluster_index(centers, last_good_center)
        if idx < len(clusters):
            selected = clusters[idx]
            last_good_center = centers[idx]
            last_good_index = idx
            for pt in selected:
                cv2.circle(frame, tuple(pt[0]), 1, (0, 255, 0), -1)

            x_fit, y_fit = ransac_polynomial_curve(selected, degree=2, residual_threshold=2)
            if x_fit is not None:
                for x, y in zip(x_fit, y_fit):
                    cv2.circle(frame, (int(x.item()), int(y.item())), radius=1, color=(0, 0, 255), thickness=-1)
                    # f 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    #     frame[y, x] = (0, 0, 255)  # 255 0 0 # 0 255 255 yellow
                cv2.circle(frame, find_lowest_point_smooth(x_fit, y_fit), radius=4, color=(255, 10, 255), thickness=-1)
        '''for i, cluster in enumerate(clusters):
            # color = colors[i % len(colors)]
            if i >= len(colors):
                colors.append(generate_new_color(colors))
            color = colors[i]
            for pt in cluster:
                cv2.circle(frame, tuple(pt[0]), 1, color, -1)'''
    return frame


class SliderWindow(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.setWindowTitle("Filter Controls")
        self.main_app = main_app
        self.sliders = {}
        layout = QGridLayout()
        for i, (name, val) in enumerate(main_app.params.items()):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(255)
            slider.setValue(val)
            slider.valueChanged.connect(main_app.update_params)
            self.sliders[name] = slider
            layout.addWidget(QLabel(name), i, 0)
            layout.addWidget(slider, i, 1)

        self.frame_selector = QSpinBox()
        self.frame_selector.setRange(0, total_frames - 1)
        self.frame_selector.valueChanged.connect(main_app.set_frame)
        layout.addWidget(QLabel("Frame"), len(main_app.params), 0)
        layout.addWidget(self.frame_selector, len(main_app.params), 1)

        self.setLayout(layout)
        self.main_app.slider_refs = self.sliders
        self.main_app.frame_selector = self.frame_selector

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welding Analysis (PyQt6)")
        self.label = QLabel(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.params = {
            'canny_min': 98,
            'canny_max': 173,
            'd': 20,
            'sigmaColor': 200,
            'sigmaSpace': 200,
            'epsilon': 8,
            'min_samples': 10,
            'leaf_size': 50
        }
        self.slider_refs = {}
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        play_btn = QPushButton("Play/Pause")
        play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(play_btn)
        self.setLayout(layout)
        self.slider_window = SliderWindow(self)
        self.slider_window.show()
        self.timer.start(30)

    def update_params(self):
        for name, slider in self.slider_refs.items():
            self.params[name] = max(1, slider.value())

    def toggle_play(self):
        global frame_paused
        frame_paused = not frame_paused

    def set_frame(self, pos):
        global frame_pos, cap
        frame_pos = pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def next_frame(self):
        global cap, frame_pos
        if frame_paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            # self.set_frame(frame_pos)
            return
        ret, frame = cap.read()
        if not ret:
            return
        processed = process_frame(frame, self.params)
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_selector.setValue(frame_pos)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.slider_window.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = VideoApp()
    win.show()
    sys.exit(app.exec())
