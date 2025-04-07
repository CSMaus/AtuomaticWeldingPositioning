import os
import cv2
from ultralytics import YOLO
import numpy as np
from helpers import compute_forward_energy, find_seam_dijkstra

vertical_energy = False


def predict_yolo(curr_frame):
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
                # energy = compute_forward_energy(gray_crop)

                h, w = energyyy.shape
                cumulative = np.copy(energyyy)
                backtrack = np.zeros_like(cumulative, dtype=np.int32)

                if vertical_energy:
                    for row in range(1, h):
                        for col in range(w):
                            min_val = cumulative[row - 1, col]
                            min_idx = col

                            if col > 0 and cumulative[row - 1, col - 1] < min_val:
                                min_val = cumulative[row - 1, col - 1]
                                min_idx = col - 1
                            if col < w - 1 and cumulative[row - 1, col + 1] < min_val:
                                min_val = cumulative[row - 1, col + 1]
                                min_idx = col + 1

                            cumulative[row, col] += min_val
                            backtrack[row, col] = min_idx
                else:
                    for col in range(1, w):
                        for row in range(h):
                            min_val = cumulative[row, col - 1]
                            min_idx = row

                            if row > 0 and cumulative[row - 1, col - 1] < min_val:
                                min_val = cumulative[row - 1, col - 1]
                                min_idx = row - 1
                            if row < h - 1 and cumulative[row + 1, col - 1] < min_val:
                                min_val = cumulative[row + 1, col - 1]
                                min_idx = row + 1

                            cumulative[row, col] += min_val
                            backtrack[row, col] = min_idx

                '''
                seam = []
                j = np.argmin(cumulative[0])  # ← from TOP, not bottom!
                seam.append((j, 0))
                for i in range(1, h):
                    j = backtrack[i, j]
                    seam.append((j, i))
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

    # if cumulative is not None:
    #     norm_cumulative = cv2.normalize(cumulative, None, 0, 255, cv2.NORM_MINMAX)
    #     energy_map_display = norm_cumulative.astype(np.uint8)
    #     cv2.imshow("Groove Energy Map", energy_map_display)

    if energyyy is not None:
        energyyy = np.abs(energyyy).astype(np.float32)
        norm_energy = cv2.normalize(energyyy, None, 0, 255, cv2.NORM_MINMAX)
        norm_energy_map_display = norm_energy.astype(np.uint8)
        cv2.imshow("Energy Map", norm_energy_map_display)

    return labeled




images_path = '/Users/kseni/Documents/GitHub/AtuomaticWeldingPositioning/CNN/datas/ds_imgs'
this_img_path = os.path.join(images_path, os.listdir(images_path)[0])
print(this_img_path)

yolo_model = YOLO("runs/segment/electrode_groove_seg2/weights/best.pt")
_ = yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)


frame = cv2.cvtColor(cv2.imread(this_img_path), cv2.COLOR_BGR2RGB)
labeled_frame = predict_yolo(frame)  # _bot_top
dis_frame = labeled_frame  # cv2.cvtColor(labeled_frame, cv2.COLOR_RGB2GRAY)
cv2.imshow("labeled frame", dis_frame)
cv2.waitKey(0)




