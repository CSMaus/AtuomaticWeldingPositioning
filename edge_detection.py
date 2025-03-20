import os
import cv2

videos_path = "/Users/kseni/Downloads/kakao/Robot REC/"
# videos_path = "D:/work_doks/projects/Doosan. Welding/2025/data/"
# this_video_path = os.path.join(videos_path, os.listdir(videos_path)[0])
this_video_path = os.path.join(videos_path, "rb_test7.mp4")
cap = cv2.VideoCapture(this_video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# placeholder for opencv
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

        if len(approx) == 3:  # Triangle shape
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
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    canny_min = cv2.getTrackbarPos("Canny Min", "Welding Analysis")
    canny_max = cv2.getTrackbarPos("Canny Max", "Welding Analysis")
    sobel_ksize = cv2.getTrackbarPos("Sobel ksize", "Welding Analysis")
    threshold_val = cv2.getTrackbarPos("Threshold", "Welding Analysis")

    sobel_ksize = max(1, sobel_ksize * 2 + 1)  # ksize is odd
    edges = cv2.Canny(blurred, canny_min, canny_max)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)

    _, thresh = cv2.threshold(sobel_x_abs, threshold_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    left_edge_x = cv2.boundingRect(sorted_contours[0])[0] if sorted_contours else None
    right_edge_x = cv2.boundingRect(sorted_contours[-1])[0] if sorted_contours else None

    electrode_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    electrode_x, electrode_y = detect_v_shaped_electrode(edges, frame)

    '''electrode_x, electrode_y = None, None
    if electrode_contours:
        electrode = max(electrode_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(electrode)
        electrode_x, electrode_y = x + w // 2, y + h // 2'''

    '''if left_edge_x:
        cv2.line(frame, (left_edge_x, 0), (left_edge_x, frame.shape[0]), (255, 0, 0), 2)
    if right_edge_x:
        cv2.line(frame, (right_edge_x, 0), (right_edge_x, frame.shape[0]), (0, 0, 255), 2)
    if electrode_x and electrode_y:
        cv2.circle(frame, (electrode_x, electrode_y), 10, (0, 255, 255), -1)
    '''


    # display edges:
    if left_edge_x:
        cv2.line(frame, (left_edge_x, 0), (left_edge_x, frame.shape[0]), (255, 0, 0), 10)
    if right_edge_x:
        cv2.line(frame, (right_edge_x, 0), (right_edge_x, frame.shape[0]), (0, 0, 255), 10)
    if electrode_x and electrode_y:
        cv2.circle(frame, (electrode_x, electrode_y), 10, (0, 255, 255), -1)

    for contour in electrode_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 10)  # green for electrode
    for contour in sorted_contours:
        cv2.drawContours(frame, [contour], -1, (255, 165, 0), 1)  # orange for weld groove edges
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

frame_paused = False
frame_pos = 0
def toggle_pause():
    global frame_paused
    frame_paused = not frame_paused

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Welding Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Welding Analysis", int(frame_width/2), int((frame_height )/1))
cv2.createTrackbar("Canny Min", "Welding Analysis", 50, 355, nothing)
cv2.createTrackbar("Canny Max", "Welding Analysis", 150, 355, nothing)
cv2.createTrackbar("Sobel ksize", "Welding Analysis", 1, 10, nothing)
cv2.createTrackbar("Threshold", "Welding Analysis", 50, 255, nothing)
cv2.createTrackbar("Frame", "Welding Analysis", 0, total_frames - 1, set_frame)

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
    if key == ord("q"):
        break
    elif key == ord("p"):
        cv2.waitKey(0)
    elif key == ord(" "):
        toggle_pause()

cap.release()
cv2.destroyAllWindows()



