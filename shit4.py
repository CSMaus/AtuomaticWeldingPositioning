import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import random
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# videos_path = "/Users/kseni/Downloads/0kakao/Robot REC/"
videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/"
# this_video_path = os.path.join(videos_path, os.listdir(videos_path)[11])
# "rb6.360mm & 30d.mp4")  # "rb_test7.mp4")
this_video_path = os.path.join(videos_path, "rb_test8.mp4")
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paused = False
frame_pos = 0
num_clusters_to_ex = 3

# ############################
cluster_history = []
MAX_HISTORY = 5
last_good_center = None
last_good_index = None
smoothed_line_x = None
all_clusters = False
previous_bottom = None

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
def smooth_vertical_line(x_pred, alpha=0.8):
    global smoothed_line_x
    x_pred = x_pred.flatten()

    if smoothed_line_x is None or len(smoothed_line_x) != len(x_pred):
        smoothed_line_x = x_pred.copy()
    else:
        smoothed_line_x = alpha * smoothed_line_x + (1 - alpha) * x_pred

    return smoothed_line_x.astype(int)

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

def get_cluster_centers(clusters):
    return [np.mean(c.reshape(-1, 2), axis=0) for c in clusters]

def find_consistent_cluster_index(cluster_centers, last_center, max_distance=40):
    if last_center is None or not cluster_centers:
        return 0  # default to first cluster initially

    best_index = None
    best_distance = float('inf')

    for i, curr in enumerate(cluster_centers):
        dist = np.linalg.norm(np.array(curr) - np.array(last_center))
        if dist < best_distance and dist < max_distance:
            best_distance = dist
            best_index = i

    return best_index if best_index is not None else 0

def nothing(x): pass
def set_frame(pos): global frame_pos; cap.set(cv2.CAP_PROP_POS_FRAMES, pos); frame_pos = pos
def toggle_pause(): global frame_paused; frame_paused = not frame_paused


def generate_new_color(existing_colors):
    while True:
        color = tuple(random.randint(70, 200) for _ in range(3))
        if color not in existing_colors:
            return color
def extract_main_cluster(edges):
    # this one extracts only one part. So, there will be contours only for electrode or only for the part below it
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) == 0:
        return np.empty((0, 1, 2), dtype=np.int32)

    points = np.stack((x_indices, y_indices), axis=1)
    clustering = DBSCAN(eps=15, min_samples=10).fit(points)

    labels = clustering.labels_
    if np.max(labels) < 0:
        return np.empty((0, 1, 2), dtype=np.int32)

    best_label = np.argmax(np.bincount(labels[labels >= 0]))
    selected = points[labels == best_label]
    return selected.reshape(-1, 1, 2)

def extract_all_clusters(edges):
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) == 0:
        return []

    points = np.stack((x_indices, y_indices), axis=1)
    epsilon = max(1, cv2.getTrackbarPos("epsilon", "Filter Controls"))
    min_samples = max(1, cv2.getTrackbarPos("min_samples", "Filter Controls"))
    leaf_size = max(1, cv2.getTrackbarPos("leaf_size", "Filter Controls"))
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples, leaf_size=leaf_size).fit(points)
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
def process_frame(frame):
    global last_good_center, last_good_index, cluster_history

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    d = max(1, cv2.getTrackbarPos("Bilateral d", "Filter Controls"))
    sigmaColor = cv2.getTrackbarPos("Bilateral SigmaColor", "Filter Controls")
    sigmaSpace = cv2.getTrackbarPos("Bilateral SigmaSpace", "Filter Controls")
    filtered = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

    canny_min = cv2.getTrackbarPos("Canny Min", "Filter Controls")
    canny_max = cv2.getTrackbarPos("Canny Max", "Filter Controls")
    edges = cv2.Canny(filtered, canny_min, canny_max)

    if num_clusters_to_ex < 2:
        cluster_points = extract_main_cluster(edges)
        if len(cluster_points) > 0:
            for pt in cluster_points:
                cv2.circle(frame, tuple(pt[0]), 1, (0, 255, 0), -1)
    else:
        if all_clusters:
            clusters = extract_all_clusters(edges)
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
            for i, cluster in enumerate(clusters):
                if i >= len(colors):
                    colors.append(generate_new_color(colors))
                color = colors[i]
                for pt in cluster:
                    cv2.circle(frame, tuple(pt[0]), 1, color, -1)
        else:
            # clusters = extract_all_clusters(edges, params['epsilon'], params['min_samples'], params['leaf_size'])
            clusters = extract_all_clusters(edges)
            centers = get_cluster_centers(clusters)
            idx = find_consistent_cluster_index(centers, last_good_center)
            if idx < len(clusters):
                selected = clusters[idx]
                last_good_center = centers[idx]
                last_good_index = idx
                for pt in selected:
                    cv2.circle(frame, tuple(pt[0]), 1, (0, 255, 0), -1)

                x_fit, y_fit = ransac_polynomial_curve(selected, degree=2, residual_threshold=5)
                if x_fit is not None:
                    for x, y in zip(x_fit, y_fit):
                        cv2.circle(frame, (int(x.item()), int(y.item())), radius=1, color=(0, 0, 255), thickness=-1)
                        # f 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        #     frame[y, x] = (0, 0, 255)  # 255 0 0 # 0 255 255 yellow
                    point = find_lowest_point_smooth(x_fit, y_fit)
                    cv2.circle(frame, point, radius=4, color=(255, 10, 10), thickness=-1)
                    cv2.putText(frame, "Electrode", (point[0], point[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 10, 10), 2)


            x_line, y_line = detect_vertical_join_line_ransac(frame.copy(), electrode_point=point,
                                                              electrode_width=50)
            if x_line is not None:
                x_line_smooth = smooth_vertical_line(x_line, alpha=0.92)
                for x, y in zip(x_line_smooth, y_line):
                    cv2.circle(frame, (int(x.item()), int(y.item())), 1, (255, 0, 255), -1)
        '''
        global last_good_center, last_good_index
        idx = find_consistent_cluster_index(centers, last_good_center)

        if idx < len(clusters):
            selected = clusters[idx]
            last_good_center = centers[idx]
            last_good_index = idx
            for pt in selected:
                cv2.circle(frame, tuple(pt[0]), 1, (0, 255, 0), -1)
        '''


    return frame


cv2.namedWindow("Welding Analysis", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", "Welding Analysis", 0, total_frames - 1, set_frame)

cv2.namedWindow("Filter Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Filter Controls", 400, 300)
cv2.createTrackbar("Canny Min", "Filter Controls", 50, 255, nothing)
cv2.createTrackbar("Canny Max", "Filter Controls", 150, 255, nothing)
cv2.createTrackbar("Bilateral d", "Filter Controls", 5, 25, nothing)
cv2.createTrackbar("Bilateral SigmaColor", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("Bilateral SigmaSpace", "Filter Controls", 75, 250, nothing)

cv2.createTrackbar("epsilon", "Filter Controls", 1, 50, nothing)
cv2.createTrackbar("min_samples", "Filter Controls", 1, 50, nothing)
cv2.createTrackbar("leaf_size", "Filter Controls", 1, 100, nothing)

def set_standart_tracks():
    cv2.setTrackbarPos("Canny Min", "Filter Controls", 98)
    cv2.setTrackbarPos("Canny Max", "Filter Controls", 173)
    cv2.setTrackbarPos("Bilateral d", "Filter Controls", 20)
    cv2.setTrackbarPos("Bilateral SigmaColor", "Filter Controls", 200)
    cv2.setTrackbarPos("Bilateral SigmaSpace", "Filter Controls", 200)
    cv2.setTrackbarPos("epsilon", "Filter Controls", 8)
    cv2.setTrackbarPos("min_samples", "Filter Controls", 10)
    cv2.setTrackbarPos("leaf_size", "Filter Controls", 50)






set_standart_tracks()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    processed_frame = process_frame(frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis", current_frame)
    cv2.imshow("Welding Analysis", processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()

cap.release()
cv2.destroyAllWindows()
