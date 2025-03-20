import cv2
import numpy as np

def apply_clahe(img, clip_limit, tile_grid_size):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_adjustments(frame, brightness, contrast, vibrance, hue, saturation, lightness, clip_limit, tile_grid_size):
    frame = apply_clahe(frame, clip_limit, tile_grid_size)
    img = np.int16(frame)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    xval = np.arange(0, 256)
    lut = (255 * np.tanh(vibrance * xval / 255) / np.tanh(1) + 0.5).astype(np.uint8)
    s = cv2.LUT(s, lut)

    h = (h.astype(int) + hue) % 180
    h = h.astype(np.uint8)

    s = cv2.add(s, saturation)
    v = cv2.add(v, lightness)

    adjusted_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
