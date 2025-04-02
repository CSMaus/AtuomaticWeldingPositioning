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
# from phasepack import phasecong

# videos_path = "/Users/kseni/Downloads/kakao/Robot REC/"
videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
this_video_path = os.path.join(videos_path, os.listdir(videos_path)[10])
# this_video_path = os.path.join(videos_path, "rb_test8.mp4")  # "rb_test7.mp4")  # "rb6.360mm & 30d.mp4"

cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_paused = False
frame_pos = 0
residual_thresh = 15
last_good_center = None
last_good_index = None

# ############################
cluster_history = []
MAX_HISTORY = 5
num_clusters_to_ex = 2

# here will be the code for the joint line detection
def detect_joint_line_fast0(gray):
    thetas = [0, np.pi/12, -np.pi/12]  # Near-vertical angles
    responses = []

    for theta in thetas:
        kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.95, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        responses.append(filtered)

    combined = np.max(np.stack(responses), axis=0)
    _, binary = cv2.threshold(combined, 250, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return binary
def detect_joint_line_fast(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    vertical_kernel = np.array([[-1], [-1], [4], [-1], [-1]], dtype=np.float32)
    line_enhanced = cv2.filter2D(enhanced, -1, vertical_kernel)

    _, binary = cv2.threshold(line_enhanced, 30, 255, cv2.THRESH_BINARY)

    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical)

    h, w = clean.shape
    center_weight = np.exp(-((np.arange(w) - w // 2) ** 2) / (2 * (w * 0.1) ** 2))
    clean = (clean * center_weight.astype(np.uint8)).astype(np.uint8)

    return clean
def detect_join_line_by_edges(edges, electrode_point, electrode_width=50):
    x_center, y_start = electrode_point
    h, w = edges.shape
    x_min = max(0, x_center - electrode_width // 2)
    x_max = min(w, x_center + electrode_width // 2)
    y_max = h

    region = edges[y_start:y_max, x_min:x_max]
    y_indices, x_indices = np.where(region > 0)

    if len(x_indices) == 0:
        return None

    points = np.stack((x_indices + x_min, y_indices + y_start), axis=1)
    clustering = DBSCAN(eps=10, min_samples=5).fit(points)
    labels = clustering.labels_

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_pts = points[labels == label]
        clusters.append(cluster_pts.reshape(-1, 1, 2))

    if not clusters:
        return None

    return max(clusters, key=lambda c: c[:, 0, 1].ptp())
def detect_join_line_by_black_column(gray, electrode_point, electrode_width=50):
    x_center, y_start = electrode_point
    h, w = gray.shape
    x_min = max(0, x_center - electrode_width // 2)
    x_max = min(w, x_center + electrode_width // 2)
    y_max = h

    region = gray[y_start:y_max, x_min:x_max]
    vertical_sums = np.sum(region, axis=0)
    darkest_col_idx = np.argmin(vertical_sums)
    x_dark = x_min + darkest_col_idx

    column = gray[y_start:y_max, x_dark]
    black_mask = column < 40
    y_coords = np.where(black_mask)[0]

    if len(y_coords) < 5:
        return None

    points = np.array([[x_dark, y + y_start] for y in y_coords]).reshape(-1, 1, 2)
    return points
def detect_vertical_join_line_ransac(original_frame, electrode_point, electrode_width=50):
    # Parameters for separate pre-processing
    bilateral_d = 3
    sigmaColor = 51
    sigmaSpace = 51
    canny_min = 98
    canny_max = 94

    x_center, y_start = electrode_point
    h, w = original_frame.shape[:2]
    x_min = max(0, x_center - electrode_width // 2)
    x_max = min(w, x_center + electrode_width // 2)
    y_max = h

    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, bilateral_d, sigmaColor, sigmaSpace)
    edges = cv2.Canny(filtered, canny_min, canny_max)

    region = edges[y_start:y_max, x_min:x_max]
    y_indices, x_indices = np.where(region > 0)

    if len(x_indices) == 0:
        return None, None

    points = np.stack((x_indices + x_min, y_indices + y_start), axis=1)

    # RANSAC vertical line fitting (x = f(y))
    if len(points) < 2:
        return None, None

    Y = points[:, 1].reshape(-1, 1)
    X = points[:, 0]
    model = make_pipeline(PolynomialFeatures(1), RANSACRegressor(residual_threshold=3, random_state=42))
    model.fit(Y, X)
    y_range = np.linspace(Y.min(), Y.max(), 200).reshape(-1, 1)
    x_pred = model.predict(y_range)

    return x_pred.astype(int), y_range.astype(int)




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
def find_lowest_point_smooth(x_fit, y_fit, alpha=0.9):
    # higher aloha - smoother
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

    # this doesn't work
    # joint_line_mask = detect_joint_line_fast(gray)
    # frame[joint_line_mask > 0] = (255, 0, 255)


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

            x_fit, y_fit = ransac_polynomial_curve(selected, degree=2, residual_threshold=residual_thresh)
            if x_fit is not None:
                for x, y in zip(x_fit, y_fit):
                    cv2.circle(frame, (int(x.item()), int(y.item())), radius=1, color=(0, 0, 255), thickness=-1)
                    # f 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    #     frame[y, x] = (0, 0, 255)  # 255 0 0 # 0 255 255 yellow
                point = find_lowest_point_smooth(x_fit, y_fit)
                cv2.circle(frame, point, radius=4, color=(255, 10, 10), thickness=-1)
                cv2.putText(frame, "Electrode", (point[0], point[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 10, 10), 2)


                # ###################################################
                # ################################################
                # Method 1: Edge-based vertical join line
                # edge_line = detect_join_line_by_edges(edges, point, electrode_width=50)
                # if edge_line is not None:
                #     for pt in edge_line:
                #         cv2.circle(frame, tuple(pt[0]), 1, (0, 255, 255), -1)

                # Method 2: Black column detection
                # black_line = detect_join_line_by_black_column(gray, point, electrode_width=50)
                # if black_line is not None:
                #     for pt in black_line:
                #         cv2.circle(frame, tuple(pt[0]), 1, (255, 255, 0), -1)

                # Method 3: vert line using edge detection below electrode and ransac
                x_line, y_line = detect_vertical_join_line_ransac(original_frame=frame.copy(), electrode_point=point,
                                                                  electrode_width=50)
                if x_line is not None:
                    for x, y in zip(x_line, y_line):
                        cv2.circle(frame, (int(x.item()), int(y.item())), 1, (255, 0, 255), -1)
                # ################################################  
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

        self.residual_threshold_selector = QSpinBox()
        self.residual_threshold_selector.setRange(1, 30)
        self.residual_threshold_selector.setValue(residual_thresh)
        self.residual_threshold_selector.valueChanged.connect(main_app.set_residual_threshold)
        layout.addWidget(QLabel("ResThresh"), len(main_app.params) + 1, 0)
        layout.addWidget(self.residual_threshold_selector, len(main_app.params) + 1, 1)

        self.setLayout(layout)
        self.main_app.slider_refs = self.sliders
        self.main_app.frame_selector = self.frame_selector
        self.main_app.residual_threshold_selector = self.residual_threshold_selector

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welding Analysis (PyQt6)")
        self.label = QLabel(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.params = {
            'canny_min': 120,
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

    def set_residual_threshold(self, value):
        global residual_thresh
        residual_thresh = value

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
