from sklearn.cluster import DBSCAN
import os
import cv2
import numpy as np
import random
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
this_video_path = os.path.join(videos_path, os.listdir(videos_path)[10]) # 11
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paused = False
frame_pos = 0

rseed = 42
np.random.seed(rseed)
random.seed(rseed)

def update_rseed(value):
    global rseed
    rseed = value
    np.random.seed(rseed)
    random.seed(rseed)

def nothing(x): pass

def split_contours_by_position(contours, y_threshold):
    upper = []
    lower = []
    for cnt in contours:
        for p in cnt:
            x, y = p[0]
            if y < y_threshold:
                upper.append(cnt)
                break
            elif y >= y_threshold:
                lower.append(cnt)
                break
    return upper, lower

def filter_electrode_contours(contours, frame_shape):
    h, w = frame_shape[:2]
    cx = w // 2

    candidate_points = []
    for cnt in contours:
        if len(cnt) < 15:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < 10 or hh < 10:
            continue
        if abs((x + ww // 2) - cx) > w * 0.15:
            continue
        if y < h * 0.1 or y > h * 0.9:
            continue
        for p in cnt:
            candidate_points.append(p[0])

    candidate_points = np.array(candidate_points)
    if len(candidate_points) == 0:
        return []

    clustering = DBSCAN(eps=50, min_samples=15).fit(candidate_points)
    labels = clustering.labels_
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    best_cluster = []
    max_count = 0
    for label in unique_labels:
        cluster_pts = candidate_points[labels == label]
        if len(cluster_pts) > max_count:
            best_cluster = cluster_pts
            max_count = len(cluster_pts)

    return best_cluster.reshape(-1, 1, 2)

def ransac_polynomial_curve_from_points(points, degree=2, residual_threshold=5):
    if len(points) < degree + 1:
        return None, None

    points = points.reshape(-1, 2)
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    model = make_pipeline(PolynomialFeatures(2),
                          RANSACRegressor(residual_threshold=residual_threshold, random_state=42))
    model.fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_range = model.predict(x_range)
    return x_range, y_range


def ransac_polynomial_curve(contours, degree=20, residual_threshold=5):
    # RANSAC with polynomial regression
    points = []
    for cnt in contours:
        for p in cnt:
            x, y = p[0]
            points.append([x, y])
    points = np.array(points)

    if len(points) < degree + 1:
        return None, None

    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=residual_threshold))
    model.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_range = model.predict(x_range)
    return x_range, y_range


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    d = max(1, cv2.getTrackbarPos("Bilateral d", "Filter Controls"))
    sigmaColor = cv2.getTrackbarPos("Bilateral SigmaColor", "Filter Controls")
    sigmaSpace = cv2.getTrackbarPos("Bilateral SigmaSpace", "Filter Controls")
    filtered = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

    canny_min = cv2.getTrackbarPos("Canny Min", "Filter Controls")
    canny_max = cv2.getTrackbarPos("Canny Max", "Filter Controls")
    edges = cv2.Canny(filtered, canny_min, canny_max)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    selected_pts = filter_electrode_contours(contours, frame.shape)
    cv2.drawContours(frame, [selected_pts], -1, (0, 0, 255), 3)

    residual_thresh = max(1, cv2.getTrackbarPos("Residual Threshold", "Filter Controls"))
    # x_fit, y_fit = ransac_polynomial_curve_from_points(selected_pts, degree=2, residual_threshold=residual_thresh)
    x_fit, y_fit = ransac_polynomial_curve_from_points(selected_pts, degree=2, residual_threshold=residual_thresh)

    if x_fit is not None:
        for x, y in zip(x_fit, y_fit):
            cv2.circle(frame, (int(x.item()), int(y.item())), 1, (0, 255, 0), -1)

    return frame


def set_frame(pos):
    global frame_pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    frame_pos = pos

def toggle_pause():
    global frame_paused
    frame_paused = not frame_paused

# UI
cv2.namedWindow("Welding Analysis2", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", "Welding Analysis2", 0, total_frames - 1, set_frame)

cv2.namedWindow("Filter Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Filter Controls", 400, 300)
cv2.createTrackbar("Canny Min", "Filter Controls", 50, 255, nothing)
cv2.createTrackbar("Canny Max", "Filter Controls", 150, 255, nothing)
cv2.createTrackbar("Bilateral d", "Filter Controls", 5, 25, nothing)
cv2.createTrackbar("Bilateral SigmaColor", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("Bilateral SigmaSpace", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("Residual Threshold", "Filter Controls", 1, 250, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    processed_frame = process_frame(frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis2", current_frame)
    cv2.imshow("Welding Analysis2", processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()

cap.release()
cv2.destroyAllWindows()

