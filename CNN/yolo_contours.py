import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN

videos_path = "D:/ML_DL_AI_stuff/!!DoosanWelding2025/data/"
video_name = "rb_test8"
this_video_path = os.path.join(videos_path, f"{video_name}.mp4")
cap = cv2.VideoCapture(this_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paused = False
frame_pos = 0

yolo_model = YOLO("runs/segment/electrode_groove_seg8/weights/best.pt")

last_groove_box = None
last_electrode_point = None
last_groove_point = None

def set_frame(pos):
    global frame_pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    frame_pos = pos

def toggle_pause():
    global frame_paused
    frame_paused = not frame_paused

def nothing(x): pass

# cv2.namedWindow("Filter Controls", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Filter Controls", 400, 100)
# cv2.createTrackbar("Canny Min", "Filter Controls", 98, 255, nothing)
# cv2.createTrackbar("Canny Max", "Filter Controls", 173, 255, nothing)

def smooth_point(new_point, last_point, alpha=0.9):
    if last_point is None:
        return new_point
    return tuple((alpha * np.array(last_point) + (1 - alpha) * np.array(new_point)).astype(int))

def draw_groove_center_point(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y


def draw_line_from_two_points(image, pt1, pt2, color=(0, 255, 255), thickness=1):
    cv2.line(image, pt1, pt2, color, thickness)

def fallback_vertical_line(image, x1, y1, x2, y2, h):
    # Fallback to nearly vertical if nothing valid detected
    margin = int(0.1 * h)
    pt1 = (x1 + (x2 - x1) // 2, y2 - margin)
    pt2 = (x1 + (x2 - x1) // 2, y1 + margin)
    draw_line_from_two_points(image, pt1, pt2)
    return True
def smooth_line_series(new_vals, axis='x', alpha=0.8):
    global smoothed_line_x, smoothed_line_y
    if axis == 'x':
        buffer = smoothed_line_x
    else:
        buffer = smoothed_line_y

    if buffer is None or len(buffer) != len(new_vals):
        buffer = new_vals.copy()
    else:
        buffer = alpha * buffer + (1 - alpha) * new_vals

    if axis == 'x':
        smoothed_line_x = buffer
    else:
        smoothed_line_y = buffer

    return buffer.astype(int)
def detect_groove_center_line_old(image, bbox, alpha=0.8):
    global smoothed_line_x
    global last_slope, last_intercept

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cmin = cv2.getTrackbarPos("Canny Min", "Filter Controls")
    cmax = cv2.getTrackbarPos("Canny Max", "Filter Controls")
    edges = cv2.Canny(gray, cmin, cmax)

    points = np.argwhere(edges > 0)
    if len(points) < 10:
        return False

    points = np.fliplr(points)
    points[:, 0] += x1
    points[:, 1] += y1

    clustering = DBSCAN(eps=8, min_samples=5).fit(points)
    points = points[clustering.labels_ != -1]
    if len(points) < 10:
        return False

    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    try:
        model = RANSACRegressor(residual_threshold=5, random_state=42)
        model.fit(X, y)

        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        if last_slope is None or last_intercept is None:
            last_slope = slope
            last_intercept = intercept
        else:
            last_slope = alpha * last_slope + (1 - alpha) * slope
            last_intercept = alpha * last_intercept + (1 - alpha) * intercept

        x_fit = np.linspace(x1, x2, 100)
        y_fit = last_slope * x_fit + last_intercept

        for x, y_val in zip(x_fit.astype(int), y_fit.astype(int)):
            if 0 <= x < image.shape[1] and 0 <= y_val < image.shape[0]:
                cv2.circle(image, (x, y_val), 1, (0, 255, 255), -1)

        '''x_fit = np.linspace(x1, x2, 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)

        y_span = y_fit.max() - y_fit.min()
        if y_span < 0.8 * h:
            y_center = (y1 + y2) // 2
            margin = int(0.4 * h)
            pt1 = (x1 + w // 2, y_center - margin)
            pt2 = (x1 + w // 2, y_center + margin)
            cv2.line(image, pt1, pt2, (0, 255, 255), 1)
            smoothed_line_x = None
            return True'''

        '''x_pred = x_fit.flatten()
        if smoothed_line_x is None or len(smoothed_line_x) != len(x_pred):
            smoothed_line_x = x_pred.copy()
        else:
            smoothed_line_x = alpha * smoothed_line_x + (1 - alpha) * x_pred
        if smoothed_line_x is not None:
            smoothed_line_x = smooth_vertical_line(smoothed_line_x, alpha=0.98)
        for x, y_val in zip(smoothed_line_x.astype(int), y_fit.astype(int)):
            if 0 <= x < image.shape[1] and 0 <= y_val < image.shape[0]:
                cv2.circle(image, (x, y_val), 1, (0, 255, 255), -1)

        smoothed_x = smooth_vertical_line(x_fit, alpha)
        for x, y_val in zip(smoothed_x, y_fit.astype(int)):
            if 0 <= x < image.shape[1] and 0 <= y_val < image.shape[0]:
                cv2.circle(image, (x, y_val), 2, (0, 255, 255), -1)'''


        return True
    except:
        return False

def detect_groove_center_line(image, bbox, alpha=0.92):
    global smoothed_line_x

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cmin = cv2.getTrackbarPos("Canny Min", "Filter Controls")
    cmax = cv2.getTrackbarPos("Canny Max", "Filter Controls")
    edges = cv2.Canny(gray, cmin, cmax)

    points = np.argwhere(edges > 0)
    if len(points) < 10:
        return False

    points = np.fliplr(points)
    points[:, 0] += x1
    points[:, 1] += y1

    clustering = DBSCAN(eps=8, min_samples=5).fit(points)
    points = points[clustering.labels_ != -1]
    if len(points) < 10:
        return False

    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    try:
        model = RANSACRegressor(residual_threshold=5, random_state=42)
        model.fit(X, y)

        x_fit = np.linspace(x1, x2, 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)

        # Check vertical coverage
        y_span = y_fit.max() - y_fit.min()
        if y_span < 0.8 * h:
            y_center = (y1 + y2) // 2
            margin = int(0.4 * h)
            pt1 = (x1 + w // 2, y_center - margin)
            pt2 = (x1 + w // 2, y_center + margin)
            cv2.line(image, pt1, pt2, (0, 255, 255), 1)
            smoothed_line_x = None
            return True

        x_line = x_fit.flatten()
        if smoothed_line_x is None or len(smoothed_line_x) != len(x_line):
            smoothed_line_x = x_line.copy()
        else:
            smoothed_line_x = alpha * smoothed_line_x + (1 - alpha) * x_line

        for x, y_val in zip(smoothed_line_x.astype(int), y_fit.astype(int)):
            if 0 <= x < image.shape[1] and 0 <= y_val < image.shape[0]:
                cv2.circle(image, (x, y_val), 1, (0, 255, 255), -1)

        return True

    except Exception as e:
        print("RANSAC error:", e)
        return False
def smooth_vertical_line(x_pred, alpha=0.98):
    global smoothed_line_x
    x_pred = x_pred.flatten()
    if smoothed_line_x is None or len(smoothed_line_x) != len(x_pred):
        smoothed_line_x = x_pred.copy()
    else:
        smoothed_line_x = alpha * smoothed_line_x + (1 - alpha) * x_pred
    return smoothed_line_x.astype(int)


def predict_yolo(curr_frame):
    global last_groove_box, last_electrode_point, last_groove_point
    labeled = curr_frame.copy()
    results = yolo_model.predict(curr_frame, verbose=False)[0]

    names = yolo_model.names
    found_groove = False
    found_electrode = False

    if results.boxes is not None:
        boxes = results.boxes

        groove_boxes = [box for box in boxes if names[int(box.cls.item())] == "groove_center"]
        if groove_boxes:
            best_groove = max(groove_boxes, key=lambda b: b.conf.item())
            xyxy = best_groove.xyxy.cpu().numpy().astype(int).flatten()
            last_groove_box = xyxy
            found_groove = True
            point = draw_groove_center_point(last_groove_box)
            smoothed_gc = smooth_point(point, last_groove_point, 0.7)
            last_groove_point = smoothed_gc
            color = (0, 255, 255)
            cv2.circle(labeled, smoothed_gc, 4, color, -1)
            cv2.putText(labeled, "groove_center", (smoothed_gc[0] + 5, smoothed_gc[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
            cls_id = int(box.cls.item())
            label = names[cls_id]

            color = (255, 0, 0) if label != "Electrode" else (50, 255, 50)
            cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            # cv2.putText(labeled, label, (xyxy[2] + 5, xyxy[3] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            """# print(label)
            if label == "groove_center":
                last_groove_box = xyxy
                found_groove = True
                # detect_groove_center_line(labeled, xyxy)
                point = draw_groove_center_point(last_groove_box)
                smoothed_gc = smooth_point(point, last_groove_point, 0.7)
                last_groove_point = smoothed_gc
                color = (0, 255, 255)
                cv2.circle(labeled, smoothed_gc, 4, color, -1)
                cv2.putText(labeled, "groove_center", (smoothed_gc[0] + 5, smoothed_gc[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1)

            el
            """
            if label == "Electrode":
                x_center = (xyxy[0] + xyxy[2]) // 2
                y_bottom = xyxy[3]
                smoothed = smooth_point((x_center, y_bottom), last_electrode_point)
                last_electrode_point = smoothed
                cv2.circle(labeled, (smoothed[0] + 5, smoothed[1]), 3, (0, 0, 255), -1)
                cv2.putText(labeled, "Electrode", (smoothed[0] + 10, smoothed[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                found_electrode = True

    if not found_groove and last_groove_box is not None:
        cv2.rectangle(labeled, (last_groove_box[0], last_groove_box[1]),
                      (last_groove_box[2], last_groove_box[3]), (255, 50, 50), 1)
        # cv2.putText(labeled, "last groove center", (last_groove_box[0], last_groove_box[1] +40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 1)

        color = (0, 255, 255)
        cv2.circle(labeled, last_groove_point, 3, color, -1)
        cv2.putText(labeled, "groove_center", (last_groove_point[0] + 5, last_groove_point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        # draw_groove_center_point(labeled, last_groove_box)
        # detect_groove_center_line(labeled, last_groove_box)

    return labeled


_ = yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

cv2.namedWindow("Welding Analysis", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Frame", "Welding Analysis", 0, total_frames - 1, set_frame)

labeled_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

    labeled_frame = predict_yolo(frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Welding Analysis", current_frame)
    cv2.imshow("Welding Analysis", labeled_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord(" "):
        toggle_pause()
    # elif key == ord("p") and frame_paused:
    #     labeled_frame = predict_yolo(frame)

cap.release()
cv2.destroyAllWindows()
