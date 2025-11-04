import cv2, numpy as np
from pypylon import pylon
from ultralytics import YOLO
from collections import deque

CLASS_GROOVE = 0
CLASS_WROD = 1

SMOOTH_BETA_X = 0.9
SMOOTH_BETA_CX = 0.93
SMOOTH_BETA_SCALE = 0.85
MAX_JUMP_MM = 0.1
MAX_JUMP_CX_MM = 0.08
MEDIAN_CX_WIN = 5


class BaslerYoloMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.3, scale_mm_per_px=None, draw_masks=True, draw_distance=True):
        self.model = YOLO(weights_path)
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        self.draw_masks = draw_masks
        self.draw_distance = draw_distance

        tlf = pylon.TlFactory.GetInstance()
        self.cam = pylon.InstantCamera(tlf.CreateFirstDevice(tlf.EnumerateDevices()[0]))
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        dummy = np.zeros((512, 512, 3), np.uint8)
        _ = self.model(dummy, verbose=False)

        self.prev_cx = None
        self.prev_xL = None
        self.prev_mm_per_px = None
        self.cx_buf = deque(maxlen=MEDIAN_CX_WIN)
        self.cx_kf = self._make_kalman_1d()

    @staticmethod
    def _make_kalman_1d():
        k = cv2.KalmanFilter(2, 1)
        k.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        k.measurementMatrix = np.array([[1, 0]], np.float32)
        k.processNoiseCov = np.eye(2, np.float32) * 1e-2
        k.measurementNoiseCov = np.array([[1e-1]], np.float32)
        k.errorCovPost = np.eye(2, np.float32)
        k.statePost = np.zeros((2, 1), np.float32)
        return k

    @staticmethod
    def _resize_masks(res, H, W):
        if res.masks is None: return []
        raw = res.masks.data.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        out = []
        for m, c in zip(raw, cls_ids):
            m8 = (m * 255).astype(np.uint8)
            m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
            out.append((c, m8))
        return out

    @staticmethod
    def _pick_largest(masks):
        if not masks: return None
        areas = [int((m > 0).sum()) for m in masks]
        return (masks[int(np.argmax(areas))] > 0).astype(np.uint8)

    @staticmethod
    def _best_wrod_bbox(res):
        if res.boxes is None: return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_ = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        best = None
        for box, c, s in zip(xyxy, cls_, conf):
            if c == CLASS_WROD and (best is None or s > best[-1]):
                best = (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(s))
        return best

    @staticmethod
    def _edge_xs_at_y(mask, y, search=6):
        H, W = mask.shape
        for dy in range(0, search + 1):
            for yy in [y] if dy == 0 else [y - dy, y + dy]:
                if 0 <= yy < H:
                    xs = np.where(mask[yy] > 0)[0]
                    if len(xs) > 0:
                        return int(xs.min()), int(xs.max()), int(yy)
        return None, None, None

    @staticmethod
    def _smooth(prev, new, beta):
        return beta * prev + (1 - beta) * new

    @staticmethod
    def _clamp_jump(prev, new, max_jump):
        if abs(new - prev) > max_jump:
            return prev + np.sign(new - prev) * max_jump
        return new

    def measure(self, conf_thresh=0.3):
        grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            return None, None

        image = self.converter.Convert(grab)
        frame = image.GetArray()
        H, W = frame.shape[:2]

        res = self.model(frame, verbose=False, conf=conf_thresh, imgsz=960, half=True)[0]
        masks = self._resize_masks(res, H, W)
        groove_masks = [m for cid, m in masks if cid == CLASS_GROOVE]
        groove_mask = self._pick_largest(groove_masks)
        ebox = self._best_wrod_bbox(res)
        if groove_mask is None or ebox is None:
            return None, frame

        x1, y1, x2, y2, _ = ebox
        cx_raw, cy = (x1 + x2) // 2, (y1 + y2) // 2
        width_px = max(1, x2 - x1)
        mm_per_px_raw = self.scale_mm_per_px_manual or (self.electrode_diameter_mm / float(width_px))
        xL_raw, _, _ = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL_raw is None:
            return None, frame

        if self.prev_mm_per_px is None:
            mm_per_px = mm_per_px_raw
        else:
            mm_per_px = self._smooth(self.prev_mm_per_px, mm_per_px_raw, SMOOTH_BETA_SCALE)

        if self.prev_cx is None:
            self.prev_cx, self.prev_xL, self.prev_mm_per_px = cx_raw, xL_raw, mm_per_px
            self.cx_buf.clear(); self.cx_buf.append(cx_raw)
            distance_mm = abs(cx_raw - xL_raw) * mm_per_px
            return distance_mm, frame

        cap_cx_px = max(1, int(round(MAX_JUMP_CX_MM / max(mm_per_px, 1e-6))))
        cap_px = max(1, int(round(MAX_JUMP_MM / max(mm_per_px, 1e-6))))

        cx_pred = float(self.cx_kf.predict()[0, 0])
        meas = np.array([[np.float32(cx_raw)]])
        cx_corr = float(self.cx_kf.correct(meas)[0, 0])
        cx_corr = self._clamp_jump(self.prev_cx, cx_corr, cap_cx_px)
        cx_s = self._smooth(self.prev_cx, cx_corr, SMOOTH_BETA_CX)
        self.cx_buf.append(cx_s)
        cx = float(np.median(self.cx_buf))

        xL_c = self._clamp_jump(self.prev_xL, xL_raw, cap_px)
        xL = self._smooth(self.prev_xL, xL_c, SMOOTH_BETA_X)

        distance_mm = abs(cx - xL) * mm_per_px

        vis = frame.copy()
        if self.draw_distance:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
            cv2.circle(vis, (int(round(xL)), int(round(cy))), 5, (255, 0, 0), -1)
            cv2.line(vis, (int(round(xL)), int(round(cy))), (int(round(cx)), int(round(cy))), (0, 255, 0), 2)
            cv2.putText(vis, f"{distance_mm:.2f} mm",
                        ((int(xL) + int(cx)) // 2, max(int(cy) - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.prev_cx, self.prev_xL, self.prev_mm_per_px = cx, xL, mm_per_px
        return distance_mm, vis

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()
