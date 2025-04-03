# inspired by  "A Real-time Passive Vision System for Robotic Arc Welding" 2015, Jinchao Liu, Zhun Fan
import os
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/"
this_video_path = os.path.join(videos_path, os.listdir(videos_path)[10]) # 11
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paused = False
frame_pos = 0

def nothing(x): pass

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


    # remove small contours which should be counted as outliners
    min_contour_len = 150 # 30
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > min_contour_len]
    cv2.drawContours(frame, filtered_contours, -1, (0, 0, 255), 3)


    r_deg = max(1, cv2.getTrackbarPos("RANSAC Degree", "Filter Controls"))
    residual_thresh = max(1, cv2.getTrackbarPos("Residual Threshold", "Filter Controls"))
    x_fit, y_fit = ransac_polynomial_curve(filtered_contours, degree=r_deg, residual_threshold=residual_thresh)

    # x_fit, y_fit = ransac_polynomial_curve(contours, degree=2, residual_threshold=5)
    # if x_fit is not None:
        # for x, y in zip(x_fit, y_fit):
            # cv2.circle(frame, (int(x.item()), int(y.item())), 1, (0, 255, 0), -1)  # green

    return frame

def set_frame(pos):
    global frame_pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    frame_pos = pos

def toggle_pause():
    global frame_paused
    frame_paused = not frame_paused

# UI
cv2.namedWindow("WeldingAnalysis", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", "WeldingAnalysis", 0, total_frames - 1, set_frame)

cv2.namedWindow("Filter Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Filter Controls", 400, 300)
cv2.createTrackbar("Canny Min", "Filter Controls", 50, 255, nothing)
cv2.createTrackbar("Canny Max", "Filter Controls", 150, 255, nothing)
cv2.createTrackbar("Bilateral d", "Filter Controls", 5, 25, nothing)
cv2.createTrackbar("Bilateral SigmaColor", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("Bilateral SigmaSpace", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("RANSAC Degree", "Filter Controls", 1, 250, nothing)
cv2.createTrackbar("Residual Threshold", "Filter Controls", 1, 250, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    processed_frame = process_frame(frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "WeldingAnalysis", current_frame)
    cv2.imshow("WeldingAnalysis", processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()

cap.release()
cv2.destroyAllWindows()


