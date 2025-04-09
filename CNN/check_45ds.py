# check that the labels for all classes are rotated correctly
# together with the images
import os
import cv2
import numpy as np

img_dir = "dataset_45/images/val"
label_dir = "dataset_45/labels/val"

class_colors = {
    0: (0, 255, 0),     # Electrode - green
    1: (255, 200, 100), # groove_center - light orange
}

def draw_yolo_segmentation(img_path, label_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3 or len(parts) % 2 == 0:
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
        color = class_colors.get(class_id, (255, 255, 255))
        if len(points) > 1:
            cv2.polylines(img, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=2)
        else:
            cv2.circle(img, points[0], radius=2, color=color, thickness=-1)

    return img

preview_count = 2
all_images = sorted(os.listdir(img_dir))[:preview_count]

for fname in all_images:
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    image_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
    img_with_mask = draw_yolo_segmentation(image_path, label_path)

    cv2.imshow("Check Rotation", img_with_mask)
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
