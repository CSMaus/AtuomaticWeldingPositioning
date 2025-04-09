import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
from helpers import smooth_groove_line


yolo_model = YOLO("runs/segment/electrode_groove_seg45/weights/best.pt")
smoothed_line_dict = {}  # key: y, value: smoothed x

def predict_yoloaa(curr_frame):
    results = yolo_model.predict(curr_frame, conf=0.0, verbose=False)  # force prediction
    labeled = curr_frame.copy()

    if results and results[0].boxes:
        names = yolo_model.names
        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            cls_id = int(box.cls.item())
            label = names[cls_id]
            cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return labeled
# was the original
def predict_yolo_bbox(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()

    names = yolo_model.names
    if results.boxes is not None:
        color_electrode = (50, 255, 50)
        color_groove = (255, 50, 150)
        did_el_bbox = False
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
            cls_id = int(box.cls.item())
            label = names[cls_id]
            if did_el_bbox:
                cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color_electrode, 2)
            else:
                cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color_groove, 2)
            cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            did_el_bbox = True

    return labeled

def predict_yolo_segm(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()

    names = yolo_model.names

    class_colors = {
        'Electrode': {'box': (0, 100, 0), 'mask': (0, 255, 0)},
        'groove_center': {'box': (100, 100, 255), 'mask': (200, 255, 255)}
    }

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            mask_color = class_colors[label]['mask']

            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled, contours, -1, mask_color, thickness=2)

    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
            cls_id = int(box.cls.item())
            label = names[cls_id]
            box_color = class_colors[label]['box']

            cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
            cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return labeled

def predict_yolo1(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()
    names = yolo_model.names

    class_colors = {
        'Electrode': {'box': (0, 100, 0), 'mask': (0, 255, 0)},
        'groove_center': {'box': (255, 100, 100), 'mask': (255, 255, 200)}
    }

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            mask_color = class_colors[label]['mask']
            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                bbox_w, bbox_h = x2 - x1, y2 - y1
                mask_cropped = mask_uint8[y1:y2, x1:x2]

                _, bin_mask = cv2.threshold(mask_cropped, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    all_pts = np.vstack(contours)
                    if len(all_pts) >= 3:
                        hull = cv2.convexHull(all_pts)

                        y_min = np.min(hull[:, 0, 1])
                        y_max = np.max(hull[:, 0, 1])
                        current_h = y_max - y_min
                        min_required_h = int(0.8 * bbox_h)
                        if current_h < min_required_h:
                            center_y = (y_max + y_min) // 2
                            new_y1 = max(0, center_y - min_required_h // 2)
                            new_y2 = min(bbox_h, center_y + min_required_h // 2)
                            scale = (new_y2 - new_y1) / max(current_h, 1)
                            hull[:, 0, 1] = ((hull[:, 0, 1] - center_y) * scale + center_y).astype(np.int32)
                            hull[:, 0, 1] = np.clip(hull[:, 0, 1], 0, bbox_h - 1)

                        hull[:, 0, 0] += x1
                        hull[:, 0, 1] += y1

                        cv2.drawContours(labeled, [hull], -1, mask_color, thickness=2)
            else:
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled, contours, -1, mask_color, thickness=2)

    drawn = set()
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        cls_id = int(box.cls.item())
        label = names[cls_id]
        box_color = class_colors[label]['box']

        if label == 'groove_center':
            key = tuple(xyxy // 10)
            if key in drawn:
                continue
            drawn.add(key)

        cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
        cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return labeled
def predict_yolo2(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()
    names = yolo_model.names

    class_colors = {
        'Electrode': {'box': (0, 100, 0), 'mask': (0, 255, 0)},
        'groove_center': {'box': (255, 100, 100), 'mask': (255, 255, 200)}
    }

    shown_classes = set()

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            if label in shown_classes:
                continue
            shown_classes.add(label)

            mask_color = class_colors[label]['mask']
            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                bbox_w, bbox_h = x2 - x1, y2 - y1
                mask_cropped = mask_uint8[y1:y2, x1:x2]

                _, bin_mask = cv2.threshold(mask_cropped, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    all_pts = np.vstack(contours)
                    if len(all_pts) >= 3:
                        hull = cv2.convexHull(all_pts)

                        y_min = np.min(hull[:, 0, 1])
                        y_max = np.max(hull[:, 0, 1])
                        current_h = y_max - y_min
                        min_required_h = int(0.8 * bbox_h)
                        if current_h < min_required_h:
                            center_y = (y_max + y_min) // 2
                            new_y1 = max(0, center_y - min_required_h // 2)
                            new_y2 = min(bbox_h, center_y + min_required_h // 2)
                            scale = (new_y2 - new_y1) / max(current_h, 1)
                            hull[:, 0, 1] = ((hull[:, 0, 1] - center_y) * scale + center_y).astype(np.int32)
                            hull[:, 0, 1] = np.clip(hull[:, 0, 1], 0, bbox_h - 1)

                        hull[:, 0, 0] += x1
                        hull[:, 0, 1] += y1
                        cv2.drawContours(labeled, [hull], -1, mask_color, thickness=2)

            elif label == 'Electrode':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled, contours, -1, mask_color, thickness=2)

    shown_classes = set()
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        cls_id = int(box.cls.item())
        label = names[cls_id]
        if label in shown_classes:
            continue
        shown_classes.add(label)

        box_color = class_colors[label]['box']
        cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
        cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return labeled

# this is better with segmentation polylines
def predict_yolo_vert_GC(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()
    names = yolo_model.names

    class_colors = {
        'Electrode': {'box': (0, 100, 0), 'mask': (0, 255, 0)},
        'groove_center': {'box': (255, 100, 100), 'mask': (255, 255, 200)}
    }

    shown_classes = set()

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            if label in shown_classes:
                continue
            shown_classes.add(label)

            mask_color = class_colors[label]['mask']
            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                mask_cropped = mask_uint8[y1:y2, x1:x2]

                ys, xs = np.where(mask_cropped > 10)
                if len(xs) > 0:
                    median_x = int(np.median(xs))
                    line_thickness = 2
                    pt1 = (x1 + median_x, y1)
                    pt2 = (x1 + median_x, y2)
                    cv2.line(labeled, pt1, pt2, mask_color, thickness=line_thickness)

            elif label == 'Electrode':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled, contours, -1, mask_color, thickness=2)

    shown_classes = set()
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        cls_id = int(box.cls.item())
        label = names[cls_id]
        if label in shown_classes:
            continue
        shown_classes.add(label)

        box_color = class_colors[label]['box']
        cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
        cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return labeled

def predict_yolo(curr_frame, electrode_width_mm, is_smooth_points=True, alpha=0.98):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()
    names = yolo_model.names

    class_colors = {
        'Electrode': {'box': (0, 100, 0), 'mask': (0, 255, 0)},
        'groove_center': {'box': (200, 100, 100), 'mask': (100, 100, 100)}
    }

    shown_classes = set()

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            if label in shown_classes:
                continue
            shown_classes.add(label)

            mask_color = class_colors[label]['mask']
            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                mask_cropped = mask_uint8[y1:y2, x1:x2]

                ys, xs = np.where(mask_cropped > 5)
                if len(xs) > 0:
                    points = []
                    mask_h = y2 - y1

                    step = max(1, mask_h // 100)

                    for rel_y in range(0, mask_h, step):
                        row = mask_cropped[rel_y, :]
                        x_vals = np.where(row > 10)[0]
                        if len(x_vals) > 0:
                            mean_x = int(np.mean(x_vals))
                            global_x = x1 + mean_x
                            global_y = y1 + rel_y
                            points.append([global_x, global_y])

                    if len(points) > 1:
                        # to be full height based on bbox
                        x_center = (x1 + x2) // 2
                        if points[0][1] > y1 + 4:
                            top_x = (points[0][0] + x_center) // 2
                            points.insert(0, [top_x, y1])
                        if points[-1][1] < y2 - 4:
                            bottom_x = (points[-1][0] + x_center) // 2
                            points.append([bottom_x, y2])

                        # to be not to curve to be close to real shape
                        # curve = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        # cv2.polylines(labeled, [curve], isClosed=False, color=mask_color, thickness=3)

                        if not is_smooth_points:
                            desired_window = 43  # odd
                            poly_order = 3
                            pts_np = np.array(points)
                            if len(pts_np) >= 5:
                                window = min(desired_window, len(pts_np) if len(pts_np) % 2 == 1 else len(pts_np) - 1)
                                smoothed_x = savgol_filter(pts_np[:, 0], window_length=window, polyorder=poly_order)
                                smoothed_points = np.stack([smoothed_x, pts_np[:, 1]], axis=1).astype(np.int32)
                            else:
                                smoothed_points = pts_np.astype(np.int32)

                        else:
                            pts_np = np.array(points, dtype=np.int32)
                            temporal_smoothed = []
                            for x, y in pts_np:
                                if y in smoothed_line_dict:
                                    prev_x = smoothed_line_dict[y]
                                    smoothed_x = alpha * prev_x + (1-alpha) * x
                                else:
                                    smoothed_x = x
                                smoothed_line_dict[y] = smoothed_x
                                temporal_smoothed.append((int(smoothed_x), y))
                            smoothed_points = np.array(temporal_smoothed, dtype=np.int32)

                        curve = smoothed_points.reshape((-1, 1, 2))
                        cv2.polylines(labeled, [curve], isClosed=False, color=mask_color, thickness=3)

                        # define pointed position:
                        top_n = 15
                        if len(smoothed_points) >= top_n:
                            top_points = smoothed_points[:top_n]
                        else:
                            top_points = smoothed_points

                        mean_x = int(np.mean(top_points[:, 0]))
                        mean_y = y1  # top bbox position
                        cv2.circle(labeled, (mean_x, mean_y), 5, (255, 250, 250), -1)
                        # to calculate distance
                        groove_center = (mean_x, mean_y)

            elif label == 'Electrode':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled, contours, -1, mask_color, thickness=2)

                # define pointed position:
                y1_box, y2_box = int(box[1]), int(box[3])
                mask_crop = mask_uint8[y1_box:y2_box, int(box[0]):int(box[2])]

                rel_y_thresh = mask_crop.shape[0] - 5  # bottom 5 pixels
                ys, xs = np.where(mask_crop > 10)
                bottom_idx = np.where(ys >= rel_y_thresh)[0]

                if len(bottom_idx) > 0:
                    xs_bot = xs[bottom_idx] + int(box[0])
                    ys_bot = ys[bottom_idx] + y1_box
                    mean_x = int(np.mean(xs_bot))
                    mean_y = int(np.mean(ys_bot))
                    cv2.circle(labeled, (mean_x, mean_y), 5, (0, 0, 255), -1)
                    # to calculate distance
                    electrode_pos = (mean_x, mean_y)
                    electrode_bbox = box.astype(int)


    shown_classes = set()
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        cls_id = int(box.cls.item())
        label = names[cls_id]
        if label in shown_classes:
            continue
        shown_classes.add(label)

        box_color = class_colors[label]['box']
        cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
        # cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        if label == 'groove_center':
            label_pos = (xyxy[2], (xyxy[1] + xyxy[3]) // 2)
            label_txt = "Center"
        else:
            label_pos = (xyxy[0], xyxy[1] - 5)
            label_txt = "Electrode"

        cv2.putText(labeled, label_txt, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    if 'groove_center' in locals() and 'electrode_pos' in locals():
        pixel_width = electrode_bbox[2] - electrode_bbox[0]
        mm_per_pixel = electrode_width_mm / pixel_width if pixel_width > 0 else 0
        dx_px = abs(groove_center[0] - electrode_pos[0])
        dx_mm = dx_px * mm_per_pixel
        display_text = f"Distance: {dx_mm:.2f} mm"
        cv2.putText(labeled, display_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    return labeled


def apply_inverse_transform(points, inv_matrix):
    points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(points_np, inv_matrix).astype(np.int32).reshape(-1, 2)

def rotate_image_with_inverse(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rot_matrix[0, 0])
    sin = abs(rot_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    inv_matrix = cv2.invertAffineTransform(rot_matrix)
    rotated_img = cv2.warpAffine(image, rot_matrix, (new_w, new_h))
    return rotated_img, rot_matrix, inv_matrix, new_w, new_h

def predict_yolo45(curr_frame, electrode_width_mm, is_smooth_points=True, alpha=0.9):
    rotated_frame, rot_matrix, inv_matrix, new_w, new_h = rotate_image_with_inverse(curr_frame, 45)
    results = yolo_model.predict(rotated_frame, verbose=False)[0]
    labeled = curr_frame.copy()
    names = yolo_model.names

    class_colors = {
        'Electrode': {'mask': (0, 255, 0)},
        'groove_center': {'mask': (230, 230, 230)}
    }
    electrode_point_color = (0, 0, 255)
    groove_c_point_color = (50, 50, 255)

    groove_center = None
    electrode_pos = None
    electrode_bbox = None

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        shown_classes = set()

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            if label in shown_classes:
                continue
            shown_classes.add(label)

            mask_color = class_colors[label]['mask']
            mask_resized = cv2.resize(mask, (new_w, new_h))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                mask_cropped = mask_uint8[y1:y2, x1:x2]

                ys, xs = np.where(mask_cropped > 5)
                if len(xs) > 0:
                    points = []
                    mask_h = y2 - y1
                    step = max(1, mask_h // 100)

                    for rel_y in range(0, mask_h, step):
                        row = mask_cropped[rel_y, :]
                        x_vals = np.where(row > 1)[0]
                        if len(x_vals) > 0:
                            mean_x = int(np.mean(x_vals))
                            global_x = x1 + mean_x
                            global_y = y1 + rel_y
                            points.append([global_x, global_y])

                    if len(points) > 1:
                        x_center = (x1 + x2) // 2
                        if points[0][1] > y1 + 4:
                            top_x = (points[0][0] + x_center) // 2
                            points.insert(0, [top_x, y1])
                        if points[-1][1] < y2 - 4:
                            bottom_x = (points[-1][0] + x_center) // 2
                            points.append([bottom_x, y2])

                        if not is_smooth_points:
                            pts_np = np.array(points)
                            if len(pts_np) >= 5:
                                window = min(43, len(pts_np) if len(pts_np) % 2 == 1 else len(pts_np) - 1)
                                smoothed_x = savgol_filter(pts_np[:, 0], window_length=window, polyorder=3)
                                smoothed_points = np.stack([smoothed_x, pts_np[:, 1]], axis=1).astype(np.int32)
                            else:
                                smoothed_points = np.array(points, dtype=np.int32)
                        else:
                            pts_np = np.array(points, dtype=np.int32)
                            temporal_smoothed = []
                            for x, y in pts_np:
                                if y in smoothed_line_dict:
                                    prev_x = smoothed_line_dict[y]
                                    smoothed_x = alpha * prev_x + (1 - alpha) * x
                                else:
                                    smoothed_x = x
                                smoothed_line_dict[y] = smoothed_x
                                temporal_smoothed.append((int(smoothed_x), y))
                            smoothed_points = np.array(temporal_smoothed, dtype=np.int32)

                        # Rotate curve back and draw
                        unrotated_curve = apply_inverse_transform(smoothed_points, inv_matrix).reshape((-1, 1, 2))
                        cv2.polylines(labeled, [unrotated_curve], isClosed=False, color=mask_color, thickness=3)

                        # Point for groove center
                        top_n = 55
                        top_points = smoothed_points[:top_n] if len(smoothed_points) >= top_n else smoothed_points
                        mean_x = int(np.mean(top_points[:, 0]))
                        median_x = int(np.median(unrotated_curve[:, 0]))
                        mean_y = y1  #####################################  mean_x
                        # unrotated_point = apply_inverse_transform([(median_x, mean_y)], inv_matrix)[0]
                        groove_center = (median_x, mean_y)  # tuple(unrotated_point)
                        cv2.circle(labeled, groove_center, 5, groove_c_point_color, -1)

            elif label == 'Electrode':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_unrotated = [apply_inverse_transform(cnt[:, 0, :], inv_matrix).reshape(-1, 1, 2) for cnt in contours]
                cv2.drawContours(labeled, contours_unrotated, -1, mask_color, thickness=2)

                y1_box, y2_box = int(box[1]), int(box[3])
                mask_crop = mask_uint8[y1_box:y2_box, int(box[0]):int(box[2])]

                ys, xs = np.where(mask_crop > 10)
                rel_y_thresh = mask_crop.shape[0] - 5
                bottom_idx = np.where(ys >= rel_y_thresh)[0]

                if len(bottom_idx) > 0:
                    xs_bot = xs[bottom_idx] + int(box[0])
                    ys_bot = ys[bottom_idx] + y1_box
                    mean_x = int(np.mean(xs_bot))
                    mean_y = int(np.mean(ys_bot))
                    unrotated_point = apply_inverse_transform([(mean_x, mean_y)], inv_matrix)[0]
                    electrode_pos = tuple(unrotated_point)
                    electrode_bbox = box.astype(int)
                    cv2.circle(labeled, electrode_pos, 5, electrode_point_color, -1)

    if groove_center and electrode_pos and electrode_bbox is not None:
        pixel_width = electrode_bbox[2] - electrode_bbox[0]
        mm_per_pixel = electrode_width_mm / pixel_width if pixel_width > 0 else 0
        dx_px = abs(groove_center[0] - electrode_pos[0])
        dx_mm = dx_px * mm_per_pixel
        cv2.putText(labeled, f"Distance: {dx_mm:.2f} mm", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    return labeled




import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
import time

# deeplab_model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
deeplab_model = models.deeplabv3_resnet50(pretrained=True)
deeplab_model.eval().cpu()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),  # (384, 384)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def predict_deeplab(frame, show_fps=False):
    input_tensor = transform(frame).unsqueeze(0).cpu()  # <-- fixed here
    start = time.time()
    with torch.no_grad():
        _ = deeplab_model(input_tensor)['out']
    end = time.time()
    if show_fps:
        print(f"Inference time: {(end - start)*1000:.1f} ms | FPS: {1 / (end - start):.2f}")


def predict_yolo_bot_top(curr_frame):
    results = yolo_model.predict(curr_frame, verbose=False)[0]
    labeled = curr_frame.copy()
    names = yolo_model.names

    class_colors = {
        'Electrode': {'box': (0, 100, 0), 'mask': (0, 255, 0)},
        'groove_center': {'box': (255, 100, 100), 'mask': (255, 255, 200)}
    }

    shown_classes = set()
    cumulative = None

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
            label = names[cls_id]
            if label in shown_classes:
                continue
            shown_classes.add(label)

            mask_color = class_colors[label]['mask']
            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                roi = curr_frame[y1:y2, x1:x2]
                gray_crop = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

                sobel_x = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
                energyyy = np.abs(sobel_x) + np.abs(sobel_y)

                h, w = energyyy.shape
                cumulative = np.copy(energyyy)
                backtrack = np.zeros_like(cumulative, dtype=np.int32)

                for row in range(h - 2, -1, -1):
                    for col in range(w):
                        left = cumulative[row + 1, col - 1] if col > 0 else float('inf')
                        down = cumulative[row + 1, col]
                        right = cumulative[row + 1, col + 1] if col < w - 1 else float('inf')

                        options = [left, down, right]
                        min_idx = np.argmin(options)
                        offset = [-1, 0, 1][min_idx]

                        backtrack[row, col] = col + offset
                        cumulative[row, col] += options[min_idx]

                '''
                seam = []
                j = np.argmin(cumulative[0])
                for i in range(h):
                    seam.append((j, i))
                    j = backtrack[i, j]
                for x, y in seam:
                    pt = (x1 + x, y1 + y)
                    cv2.circle(labeled, pt, 1, (0, 0, 255), 3)
                '''
                '''
                seam = find_seam_dijkstra(energy)
                for x, y in seam:
                    pt = (x1 + x, y1 + y)
                    cv2.circle(labeled, pt, 1, (0, 255, 0), 3)
                '''
                for row in range(h):
                    col = np.argmin(cumulative[row])
                    pt = (x1 + col, y1 + row)
                    cv2.circle(labeled, pt, 1, (0, 255, 0), 3)

            elif label == 'Electrode':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled, contours, -1, mask_color, thickness=2)

    shown_classes = set()
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        cls_id = int(box.cls.item())
        label = names[cls_id]
        if label in shown_classes:
            continue
        shown_classes.add(label)

        box_color = class_colors[label]['box']
        cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
        cv2.putText(labeled, label, (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    if cumulative is not None:
        norm_cumulative = cv2.normalize(cumulative, None, 0, 255, cv2.NORM_MINMAX)
        energy_map_display = norm_cumulative.astype(np.uint8)
        cv2.imshow("Groove Energy Map", energy_map_display)

    return labeled

