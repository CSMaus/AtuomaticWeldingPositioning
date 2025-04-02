import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import random


# videos_path = "/Users/kseni/Downloads/kakao/Robot REC/"
videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
this_video_path = os.path.join(videos_path, os.listdir(videos_path)[10])
# this_video_path = os.path.join(videos_path, "rb6.360mm & 30d.mp4")  # "rb_test7.mp4")
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


def process_frame(frame):
    global cluster_history

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
        clusters = extract_all_clusters(edges)
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
        for i, cluster in enumerate(clusters):
            if i >= len(colors):
                colors.append(generate_new_color(colors))
            color = colors[i]
            for pt in cluster:
                cv2.circle(frame, tuple(pt[0]), 1, color, -1)
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
