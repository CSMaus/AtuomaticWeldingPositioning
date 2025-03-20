import os
import cv2
from light_enhance import *

videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
this_video_path = os.path.join(videos_path, os.listdir(videos_path)[1])
# this_video_path = os.path.join(videos_path, "rb6.360mm & 30d.mp4")  # "rb_test7.mp4")
cap = cv2.VideoCapture(this_video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# PARAMS:
frame_paused = False
frame_pos = 0
doCLAHE_light = False

def nothing(x):
    pass

def detect_v_shaped_electrode(edges, frame):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    v_candidates = []

    height, width = frame.shape[:2]
    center_x = width // 2
    min_size = 10

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(contour)

            if h > min_size and w > min_size and abs(x + w // 2 - center_x) < 100:
                v_candidates.append((contour, y + h))

    if v_candidates:
        v_candidates.sort(key=lambda x: x[1])
        best_v = v_candidates[0][0]
        x, y, w, h = cv2.boundingRect(best_v)
        electrode_x, electrode_y = x + w // 2, y + h

        cv2.drawContours(frame, [best_v], -1, (0, 255, 255), 2)
        cv2.circle(frame, (electrode_x, electrode_y), 10, (0, 255, 255), -1)

        return electrode_x, electrode_y
    return None, None

def clahe_light_frame(f):
    brightness = cv2.getTrackbarPos("Brightness", "Brightness & CLAHE Controls")
    contrast = cv2.getTrackbarPos("Contrast", "Brightness & CLAHE Controls")
    vibrance = cv2.getTrackbarPos("Vibrance", "Brightness & CLAHE Controls") / 10
    hue = cv2.getTrackbarPos("Hue", "Brightness & CLAHE Controls")
    saturation = cv2.getTrackbarPos("Saturation", "Brightness & CLAHE Controls")
    lightness = cv2.getTrackbarPos("Lightness", "Brightness & CLAHE Controls")
    clip_limit = cv2.getTrackbarPos("CLAHE Clip Limit", "Brightness & CLAHE Controls") / 10
    tile_grid_size = max(1, cv2.getTrackbarPos("CLAHE Tile Grid Size", "Brightness & CLAHE Controls"))

    adjusted_frame = apply_adjustments(f,
                                       brightness,
                                       contrast,
                                       vibrance, hue, saturation,
                                       lightness,
                                       clip_limit,
                                       int(tile_grid_size))
    return adjusted_frame
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    d = max(1, cv2.getTrackbarPos("Bilateral d", "Filter Controls"))
    sigmaColor = cv2.getTrackbarPos("Bilateral SigmaColor", "Filter Controls")
    sigmaSpace = cv2.getTrackbarPos("Bilateral SigmaSpace", "Filter Controls")

    filtered = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

    canny_min = cv2.getTrackbarPos("Canny Min", "Filter Controls")
    canny_max = cv2.getTrackbarPos("Canny Max", "Filter Controls")
    sobel_ksize = cv2.getTrackbarPos("Sobel ksize", "Filter Controls")
    threshold_val = cv2.getTrackbarPos("Threshold", "Filter Controls")

    sobel_ksize = max(1, sobel_ksize * 2 + 1)
    edges = cv2.Canny(filtered, canny_min, canny_max)
    sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)

    _, thresh = cv2.threshold(sobel_x_abs, threshold_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    left_edge_x = cv2.boundingRect(sorted_contours[0])[0] if sorted_contours else None
    right_edge_x = cv2.boundingRect(sorted_contours[-1])[0] if sorted_contours else None

    electrode_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # electrode_x, electrode_y = detect_v_shaped_electrode(edges, frame)
    electrode_x, electrode_y = None, None
    if electrode_contours:
        electrode = max(electrode_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(electrode)
        electrode_x, electrode_y = x + w // 2, y + h

    if left_edge_x:
        cv2.line(frame, (left_edge_x, 0), (left_edge_x, frame.shape[0]), (255, 0, 0), 10)
    if right_edge_x:
        cv2.line(frame, (right_edge_x, 0), (right_edge_x, frame.shape[0]), (0, 0, 255), 10)
    if electrode_x and electrode_y:
        cv2.circle(frame, (electrode_x, electrode_y), 10, (0, 255, 255), -1)

    for contour in electrode_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 10)
    for contour in sorted_contours:
        cv2.drawContours(frame, [contour], -1, (255, 165, 0), 1)


    if electrode_x is not None and left_edge_x is not None and right_edge_x is not None:
        left_distance = electrode_x - left_edge_x
        right_distance = right_edge_x - electrode_x
        cv2.putText(frame, f"L: {left_distance} px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"R: {right_distance} px", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

def set_frame(pos):
    global frame_pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    frame_pos = pos


def toggle_pause():
    global frame_paused
    frame_paused = not frame_paused

def toggle_clahe_light():
    global doCLAHE_light
    doCLAHE_light = not doCLAHE_light

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Welding Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Welding Analysis", frame_width, frame_height)  #  // 2

cv2.namedWindow("Filter Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Filter Controls", 400, 300)

cv2.createTrackbar("Canny Min", "Filter Controls", 50, 255, nothing)
cv2.createTrackbar("Canny Max", "Filter Controls", 150, 255, nothing)
cv2.createTrackbar("Sobel ksize", "Filter Controls", 1, 10, nothing)
cv2.createTrackbar("Threshold", "Filter Controls", 50, 255, nothing)
cv2.createTrackbar("Bilateral d", "Filter Controls", 5, 25, nothing)
cv2.createTrackbar("Bilateral SigmaColor", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("Bilateral SigmaSpace", "Filter Controls", 75, 250, nothing)
cv2.createTrackbar("Frame", "Welding Analysis", 0, total_frames - 1, set_frame)

clahe_window_name = "Brightness & CLAHE Controls"
cv2.namedWindow(clahe_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(clahe_window_name, 400, 300)
cv2.createTrackbar("Brightness", clahe_window_name, 50, 100, nothing)
cv2.createTrackbar("Contrast", clahe_window_name, 50, 100, nothing)
cv2.createTrackbar("Vibrance", clahe_window_name, 15, 30, nothing)
cv2.createTrackbar("Hue", clahe_window_name, 0, 180, nothing)
cv2.createTrackbar("Saturation", clahe_window_name, 50, 100, nothing)
cv2.createTrackbar("Lightness", clahe_window_name, 50, 100, nothing)
cv2.createTrackbar("CLAHE Clip Limit", clahe_window_name, 50, 100, nothing)
cv2.createTrackbar("CLAHE Tile Grid Size", clahe_window_name, 12, 42, nothing)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    if doCLAHE_light:
        frame = clahe_light_frame(frame)
    processed_frame = process_frame(frame)

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis", current_frame)
    cv2.imshow("Welding Analysis", processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        cv2.waitKey(0)
    elif key == ord(" "):
        toggle_pause()
    elif key == ord("C") and cv2.EVENT_FLAG_SHIFTKEY:
        toggle_clahe_light()
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
