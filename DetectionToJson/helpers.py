import cv2
import numpy as np
import json
import os
import builtins
# TODO: display L-groove and R-Groove
# TODO: display groove masks as contours
# TODO: calculate the distance between groove and rod


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


def get_masks_points_distance45(curr_frame,
                              electrode_width_mm,
                              yolo_model,
                              camera_rotation_angle=0,
                              is_smooth_points=False,
                              alpha=0.9):
    """
        Predict masks of electrode and groove center and calculate
        their positions as points, and distance (yolo11n-seg)

        The original YOLO 11 model was trained on images rotated by 45 degrees,
        bcs if the groove center is fully vertical bbox width becomes 0,
        so the groove center would not be displayed.

        :param curr_frame: input frame. Each frame processed independently
        :param electrode_width_mm: width of the electrode in mm to calculate distance. The camera distortions are not noted.
        :param yolo_model: the model trained to use rotated 45 degrees images
        :param camera_rotation_angle: the angle of camera rotation, i e angle of the electrode rotation. To calculate the distance
        :param is_smooth_points: smooth or not the points of the groove center (not good enough now)
        :param alpha: if is_smooth_points is True, the alpha value for smoothing based on previous positions
        :return: segmentation masks and points of the groove center and electrode
        """
    # TODO: rotate the curr_frame based on camera_rotation_angle to make it normal. i e if camera rotation angle is 0,
    # then it should be as it is, otherwise we have to rotate it to the needed angle
    # curr_frame =
    # and after rotating original frame we can adjust its 45 degree rotation

    # rotated_frame, rot_matrix, inv_matrix, new_w, new_h = rotate_image_with_inverse(curr_frame, 45)
    pre_rotated_frame, pre_rot_matrix, pre_inv_matrix, pre_w, pre_h = rotate_image_with_inverse(curr_frame, camera_rotation_angle)
    rotated_frame, rot_matrix, inv_matrix, new_w, new_h = rotate_image_with_inverse(pre_rotated_frame, 45)

    results = yolo_model.predict(rotated_frame, verbose=False)[0]

    groove_center_points, electrode_contours, groove_center_point, electrode_point, electrode_bbox = None, None, None, None, None
    smoothed_line_dict = {}
    pixel_width_mask = 0

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        names = yolo_model.names
        for mask, box, cls_id in zip(masks, boxes, classes):
            label = names[cls_id]
            mask_resized = cv2.resize(mask, (new_w, new_h))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            if label == 'groove_center':
                x1, y1, x2, y2 = box.astype(int)
                mask_cropped = mask_uint8[y1:y2, x1:x2]
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
                    pts_np = np.array(points, dtype=np.int32)
                    if not is_smooth_points:
                        # window = min(43, len(pts_np) if len(pts_np) % 2 == 1 else len(pts_np) - 1)
                        # smoothed_x = savgol_filter(pts_np[:, 0], window_length=window, polyorder=3)
                        # smoothed_points = np.stack([smoothed_x, pts_np[:, 1]], axis=1).astype(np.int32)
                        smoothed_points = pts_np
                    else:
                        temporal_smoothed = []
                        for x, y in pts_np:
                            prev_x = smoothed_line_dict.get(y, x)
                            smoothed_x = alpha * prev_x + (1 - alpha) * x
                            smoothed_line_dict[y] = smoothed_x
                            temporal_smoothed.append((int(smoothed_x), y))
                        smoothed_points = np.array(temporal_smoothed, dtype=np.int32)


                    groove_center_points = apply_inverse_transform(smoothed_points, inv_matrix).reshape((-1, 1, 2))
                    median_x = int(np.median(groove_center_points[:, 0, 0]))
                    mean_y = int(np.mean(groove_center_points[:, 0, 1]))
                    groove_center_point = (median_x, mean_y)

                    # rotate all back to display contours correctly after rotating frame to camera angle
                    groove_center_points = apply_inverse_transform(groove_center_points, pre_inv_matrix).reshape((-1, 1, 2))

            elif label == 'Electrode':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                electrode_contours = [apply_inverse_transform(cnt[:, 0, :], inv_matrix).reshape(-1, 1, 2) for cnt in contours]

                y1_box, y2_box = int(box[1]), int(box[3])
                mask_crop = mask_uint8[y1_box:y2_box, int(box[0]):int(box[2])]
                ys, xs = np.where(mask_crop > 10)
                if len(ys) == 0 or len(xs) == 0:
                    print("Electrode mask is empty, skipping this contour.")
                    continue

                rel_y_thresh = mask_crop.shape[0] - 5
                bottom_idx = np.where(ys >= rel_y_thresh)[0]

                if len(bottom_idx) > 0:
                    xs_bot = xs[bottom_idx] + int(box[0])
                    ys_bot = ys[bottom_idx] + y1_box
                    mean_x = int(np.mean(xs_bot))
                    mean_y = int(np.mean(ys_bot))
                    electrode_point = tuple(apply_inverse_transform([(mean_x, mean_y)], inv_matrix)[0])
                    electrode_bbox = box.astype(int)

                # pixel_width = electrode_bbox[2] - electrode_bbox[0], which is okay if no rotation
                if len(ys) > 0:
                    top_y = np.min(ys)
                    bottom_y = np.max(ys)
                    top_half_y = top_y + (bottom_y - top_y) // 2

                    widths = []
                    for y in range(top_y + 3, top_half_y):
                        x_this_y = xs[ys == y]
                        if len(x_this_y) > 1:
                            width = x_this_y.max() - x_this_y.min()
                            widths.append(width)
                    if len(widths) > 0:
                        pixel_width_mask = np.median(widths)
                    else:
                        pixel_width_mask = None
                else:
                    pixel_width_mask = None

                # rotate all back to display contours correctly after rotating frame to camera angle
                electrode_contours = [apply_inverse_transform(cnt[:, 0, :], pre_inv_matrix).reshape(-1, 1, 2) for cnt in electrode_contours]

    distance_mm = None
    if groove_center_point and electrode_point and electrode_bbox is not None:
        # pixel_width = electrode_bbox[2] - electrode_bbox[0]
        # mm_per_pixel = electrode_width_mm / pixel_width if pixel_width > 0 else 0
        if pixel_width_mask is not None and pixel_width_mask !=0:
            mm_per_pixel = electrode_width_mm / pixel_width_mask
        else:
            mm_per_pixel = 0

        dx_px = abs(groove_center_point[0] - electrode_point[0])
        # rotate them back only after calculating distance
        groove_center_point = tuple(apply_inverse_transform([groove_center_point], pre_inv_matrix)[0])
        electrode_point = tuple(apply_inverse_transform([electrode_point], pre_inv_matrix)[0])
        raw_dx_mm = dx_px * mm_per_pixel
        distance_mm = raw_dx_mm * np.cos(np.radians(camera_rotation_angle)) # TODO: need to test. I'm not sure about it



    return {
        'groove_center_points': groove_center_points,
        'electrode_contours': electrode_contours,
        'groove_center_point': groove_center_point,
        'electrode_point': electrode_point,
        'distance_mm': distance_mm,
        'bboxes': None
    }

def get_masks_points_distance(curr_frame,
                              electrode_width_mm,
                              yolo_model,
                              camera_rotation_angle=0,
                              is_smooth_points=False,
                              alpha=0.9):
    """
        Predict masks of electrode and groove center and calculate
        their positions as points, and distance (yolo11n-seg)

        The original YOLO 11 model was trained on images without rotation,
        so sometimes when the groove center is fully vertical bbox width becomes 0,
        and the groove center would not be displayed.

        :param curr_frame: input frame. Each frame processed independently
        :param electrode_width_mm: width of the electrode in mm to calculate distance. The camera distortions are not noted.
        :param yolo_model: the model trained to use rotated 45 degrees images
        :param camera_rotation_angle: the angle of camera rotation, i e angle of the electrode rotation. To calculate the distance
        :param is_smooth_points: smooth or not the points of the groove center (not good enough now)
        :param alpha: if is_smooth_points is True, the alpha value for smoothing based on previous positions
        :return: segmentation masks and points of the groove center and electrode
        """
    # TODO: rotate the curr_frame based on camera_rotation_angle to make it normal. i e if camera rotation angle is 0,
    # then it should be as it is, otherwise we have to rotate it to the needed angle
    # curr_frame =
    curr_frame, rot_matrix, inv_matrix, new_w, new_h = rotate_image_with_inverse(curr_frame, camera_rotation_angle)

    results = yolo_model.predict(curr_frame, verbose=False)[0]

    groove_center_points, electrode_contours, groove_center_point, electrode_point, electrode_bbox = None, None, None, None, None
    bboxes = []
    groove_masks = []
    groove_bboxes = []
    smoothed_line_dict = {}
    pixel_width_mask = 0

    if results.masks is not None and results.boxes is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        names = yolo_model.names
        for mask, box, cls_id in zip(masks, boxes, classes):
            label = names[cls_id]
            mask_resized = cv2.resize(mask, (curr_frame.shape[1], curr_frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            bbox = box.astype(int)
            bboxes.append({'label': label, 'bbox': bbox.tolist()})

            if label == 'groove_center':
                x1, y1, x2, y2 = bbox
                mask_cropped = mask_uint8[y1:y2, x1:x2]
                ys, xs = np.where(mask_cropped > 5)
                points = []

                if len(xs) > 0:
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
                        pts_np = np.array(points, dtype=np.int32)
                        smoothed_points = pts_np if not is_smooth_points else np.array(
                            [(int(alpha * smoothed_line_dict.get(y, x) + (1 - alpha) * x), y)
                             for x, y in pts_np], dtype=np.int32)

                        groove_center_points = smoothed_points.reshape((-1, 1, 2))
                        mean_x = int(np.mean(groove_center_points[:, 0, 0]))
                        mean_y = int(np.mean(groove_center_points[:, 0, 1]))
                        groove_center_point = (mean_x, mean_y)

            elif label == 'Electrode' or label == 'W-Rod':
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                electrode_contours = contours

                y1_box, y2_box = bbox[1], bbox[3]
                mask_crop = mask_uint8[y1_box:y2_box, bbox[0]:bbox[2]]
                ys, xs = np.where(mask_crop > 10)
                bottom_idx = np.where(ys >= mask_crop.shape[0] - 5)[0]

                if len(bottom_idx) > 0:
                    mean_x = int(np.mean(xs[bottom_idx]) + bbox[0])
                    mean_y = int(np.mean(ys[bottom_idx]) + y1_box)
                    electrode_point = (mean_x, mean_y)
                    electrode_bbox = bbox
            else:
                bbox = box.astype(int)
                x1, y1, x2, y2 = bbox
                groove_bboxes.append(bbox)
                groove_masks.append(mask_uint8[y1:y2, x1:x2].copy())


    groove_relative_label = None
    if len(groove_bboxes) > 0 and electrode_bbox is not None:
        groove_x = (groove_bboxes[0][0] + groove_bboxes[0][2]) // 2
        wrod_x = (electrode_bbox[0] + electrode_bbox[2]) // 2
        groove_relative_label = "L-Gro" if groove_x < wrod_x else "R-Gro"

    distance_mm = None
    if groove_center_point and electrode_point and electrode_bbox is not None:
        pixel_width = electrode_bbox[2] - electrode_bbox[0]
        mm_per_pixel = electrode_width_mm / pixel_width if pixel_width > 0 else 0
        dx_px = abs(groove_center_point[0] - electrode_point[0])
        raw_dx_mm = dx_px * mm_per_pixel
        distance_mm = raw_dx_mm * np.cos(np.radians(camera_rotation_angle))

    # rotate all back to display contours correctly after rotating frame to camera angle
    if groove_center_points is not None:
        groove_center_points = apply_inverse_transform(groove_center_points, inv_matrix).reshape((-1, 1, 2))
    if electrode_contours is not None:
        electrode_contours = [apply_inverse_transform(cnt[:, 0, :], inv_matrix).reshape(-1, 1, 2) for cnt in electrode_contours]
    if electrode_point is not None:
        electrode_point = tuple(apply_inverse_transform([electrode_point], inv_matrix)[0])
    if groove_center_point is not None:
        groove_center_point = tuple(apply_inverse_transform([groove_center_point], inv_matrix)[0])

    return {
        'groove_center_points': groove_center_points,
        'electrode_contours': electrode_contours,
        'groove_center_point': groove_center_point,
        'groove_bboxes': groove_bboxes,
        'groove_relative_label': groove_relative_label,
        'groove_masks': groove_masks,
        'electrode_point': electrode_point,
        'distance_mm': distance_mm,
        'bboxes': bboxes
    }



def draw_masks_points_distance_0(curr_frame, prediction_output,
                               camera_rotation_angle=0, is_draw_masks=True,
                               is_draw_distance=True,
                               isuseNewLabels=True):
    # TODO: rotate the curr_frame based on camera_rotation_angle to make it normal. i e if camera rotation angle is 0,
    # then it should be as it is, otherwise we have to rotate it to the needed angle
    # labeled is rotated curr frame
    labeled = curr_frame.copy()

    wrod_lbl = "W-Rod"
    groove_lbl = "Groove"
    if not isuseNewLabels:
        wrod_lbl = "W-rod"
        groove_lbl = "Center"

    class_colors = {
        'Electrode': {'mask': (0, 255, 0)},
        'groove_center': {'mask': (230, 230, 230)},
        'Groove': {'mask': (0, 230, 230)}
    }
    electrode_point_color = (0, 0, 255)
    groove_c_point_color = (50, 50, 255)

    if is_draw_masks:
        if prediction_output['groove_center_points'] is not None:
            cv2.polylines(labeled, [prediction_output['groove_center_points']],
                          isClosed=False, color=class_colors['groove_center']['mask'], thickness=3)

        if prediction_output['electrode_contours'] is not None:
            cv2.drawContours(labeled, prediction_output['electrode_contours'],
                             -1, class_colors['Electrode']['mask'], thickness=2)

        if prediction_output['groove_masks'] is not None:
            for mask, bbox in zip(prediction_output['groove_masks'], prediction_output['groove_bboxes']):
                x1, y1, x2, y2 = bbox
                labeled[y1:y2, x1:x2][mask > 10] = class_colors['Groove']['mask']
                cv2.rectangle(labeled, (x1, y1), (x2, y2), class_colors['Groove']['mask'], thickness=2)


        # if is_draw_points:
        if prediction_output['groove_center_point']:
            cv2.circle(labeled, prediction_output['groove_center_point'], 5, groove_c_point_color, -1)

        if prediction_output['electrode_point']:
            cv2.circle(labeled, prediction_output['electrode_point'], 5, electrode_point_color, -1)

        if prediction_output['bboxes'] is not None and camera_rotation_angle == 0:
            colors = {'Electrode': (0, 100, 0), 'groove_center': (200, 100, 100)}
            drawn_labels = set()
            for bbox_info in prediction_output['bboxes']:
                label = bbox_info['label']
                if label == 'W-Rod':
                    if label in drawn_labels:
                        continue

                    drawn_labels.add(label)
                    x1, y1, x2, y2 = bbox_info['bbox']

                    if label == 'groove_center' or label == 'Groove':
                        label_pos = (x2, (y1 + y2) // 2)
                        label_txt = groove_lbl
                    else:
                        label_pos = (x1, y1 - 5)
                        label_txt = wrod_lbl
                    cv2.rectangle(labeled, (x1, y1), (x2, y2), colors.get(label, (0, 200, 0)), 2)
                    cv2.putText(labeled, label_txt, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colors.get(label, (200, 150, 150)), 2)
    return labeled

def draw_masks_points_distance(curr_frame, prediction_output,
                               camera_rotation_angle=0,
                               is_draw_masks=True,
                               is_draw_distance=True,
                               is_draw_groove_masks=True,
                               isuseNewLabels=True):
    labeled = curr_frame.copy()

    wrod_lbl = "W-Rod"
    groove_lbl = prediction_output.get('groove_relative_label', "Groove")
    if not isuseNewLabels:
        wrod_lbl = "W-rod"
        groove_lbl = "Center"

    class_colors = {
        'Electrode': {'mask': (0, 255, 0)},
        'groove_center': {'mask': (230, 230, 230)},
        'Groove': {'mask': (0, 230, 230)}
    }
    electrode_point_color = (0, 0, 255)
    groove_c_point_color = (50, 50, 255)

    if is_draw_masks:
        if prediction_output['groove_center_points'] is not None:
            cv2.polylines(labeled, [prediction_output['groove_center_points']],
                          isClosed=False, color=class_colors['groove_center']['mask'], thickness=3)

        if prediction_output['electrode_contours'] is not None and prediction_output.get('bboxes') is not None:
            for bbox_info in prediction_output['bboxes']:
                if bbox_info['label'] == 'W-Rod':
                    x1, y1, x2, y2 = bbox_info['bbox']
                    mask_box = np.zeros_like(labeled[:, :, 0], dtype=np.uint8)
                    for cnt in prediction_output['electrode_contours']:
                        cv2.drawContours(mask_box, [cnt], -1, 255, -1)
                    mask_box[:y1] = 0
                    mask_box[y2:] = 0
                    mask_box[:, :x1] = 0
                    mask_box[:, x2:] = 0
                    labeled[mask_box > 0] = class_colors['Electrode']['mask']

        if is_draw_groove_masks and prediction_output['groove_masks'] is not None:
            for mask, bbox in zip(prediction_output['groove_masks'], prediction_output['groove_bboxes']):
                x1, y1, x2, y2 = bbox
                labeled[y1:y2, x1:x2][mask > 10] = class_colors['Groove']['mask']
                cv2.rectangle(labeled, (x1, y1), (x2, y2), class_colors['Groove']['mask'], thickness=2)

        if prediction_output['groove_center_point']:
            cv2.circle(labeled, prediction_output['groove_center_point'], 5, groove_c_point_color, -1)

        if prediction_output['electrode_point']:
            cv2.circle(labeled, prediction_output['electrode_point'], 5, electrode_point_color, -1)

        if prediction_output['bboxes'] is not None and camera_rotation_angle == 0:
            colors = {'Electrode': (0, 100, 0), 'groove_center': (200, 100, 100)}
            for bbox_info in prediction_output['bboxes']:
                label = bbox_info['label']
                if label == 'groove_center' or label == 'Groove':
                    x1, y1, x2, y2 = bbox_info['bbox']
                    label_pos = (x2, (y1 + y2) // 2)
                    cv2.rectangle(labeled, (x1, y1), (x2, y2), colors.get(label, (0, 200, 0)), 2)
                    cv2.putText(labeled, groove_lbl, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colors.get(label, (200, 150, 150)), 2)
                elif label == 'W-Rod':
                    x1, y1, x2, y2 = bbox_info['bbox']
                    label_pos = (x1, y1 - 5)
                    cv2.rectangle(labeled, (x1, y1), (x2, y2), colors.get(label, (0, 200, 0)), 2)
                    cv2.putText(labeled, wrod_lbl, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colors.get(label, (200, 150, 150)), 2)

    return labeled

def write_json_file(json_data, json_file_name):
    '''data = json_data.copy()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()'''

    def safe_convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, tuple):
            return [safe_convert(x) for x in obj]
        elif isinstance(obj, list):
            return [safe_convert(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: safe_convert(v) for k, v in obj.items()}
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # fallback to string if unknown type (prevent freezing)
            return str(obj)

    data = safe_convert(json_data)

    final_filename = f"{json_file_name}.json"
    with open(final_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
        # json_file.flush()
        # os.fsync(json_file.fileno())


