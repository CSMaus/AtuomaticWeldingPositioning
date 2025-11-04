# more stability- removes a lot of fluctuations
# BaslerYoloMeasurer.py
import cv2, numpy as np
from pypylon import pylon
from ultralytics import YOLO
from collections import deque

# --- classes from your model ---
CLASS_GROOVE = 0
CLASS_WROD   = 1

# --- smoothing / stability params (same semantics as in (2)) ---
SMOOTH_BETA_X     = 0.90
SMOOTH_BETA_CX    = 0.93
SMOOTH_BETA_SCALE = 0.85
MAX_JUMP_MM       = 0.10   # cap small jitter in mm for xL
MAX_JUMP_CX_MM    = 0.08   # cap small jitter in mm for cx
MEDIAN_CX_WIN     = 5      # median window for cx
CONF_MARGIN       = 0.20   # for mask selection if you later switch to multi-groove

class Debounce1Px:
    """Blocks single-pixel flicker unless it persists N frames."""
    def __init__(self, deadband_px=1, persist_frames=3):
        self.db = int(deadband_px)
        self.N  = int(persist_frames)
        self.hold_val = None
        self.hold_cnt = 0

    def step(self, prev_committed, candidate):
        # accept immediately if outside deadband
        if prev_committed is None:
            return float(candidate)
        if abs(candidate - prev_committed) > self.db:
            self.hold_val = None; self.hold_cnt = 0
            return float(candidate)
        # inside deadband → require persistence
        if self.hold_val is None or abs(candidate - self.hold_val) > 0.5:
            self.hold_val = float(candidate); self.hold_cnt = 1
            return float(prev_committed)
        self.hold_cnt += 1
        if self.hold_cnt >= self.N:
            self.hold_val = None; self.hold_cnt = 0
            return float(candidate)
        return float(prev_committed)

class BaslerYoloMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.3, scale_mm_per_px=None,
                 draw_masks=True, draw_distance=True):
        # --- model ---
        self.model = YOLO(weights_path)
        # (GPU/half optional; keep default for widest compatibility)
        dummy = np.zeros((512, 512, 3), np.uint8)
        _ = self.model(dummy, verbose=False)  # warm-up

        # --- camera ---
        tlf = pylon.TlFactory.GetInstance()
        self.cam = pylon.InstantCamera(tlf.CreateFirstDevice(tlf.EnumerateDevices()[0]))
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # --- config ---
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        self.draw_masks = draw_masks
        self.draw_distance = draw_distance

        # --- state for smoothing ---
        self.prev_cx = None
        self.prev_xL = None
        self.prev_mm_per_px = None
        self.cx_buf = deque(maxlen=MEDIAN_CX_WIN)
        self.cx_kf  = self._make_kalman_1d()
        self.cx_db  = Debounce1Px(deadband_px=1, persist_frames=3)
        self.xL_db  = Debounce1Px(deadband_px=1, persist_frames=3)

    # -------------- helpers --------------
    @staticmethod
    def _make_kalman_1d():
        k = cv2.KalmanFilter(2, 1)
        k.transitionMatrix    = np.array([[1, 1], [0, 1]], np.float32)
        k.measurementMatrix   = np.array([[1, 0]], np.float32)
        k.processNoiseCov     = np.eye(2, np.float32) * 1e-2
        k.measurementNoiseCov = np.array([[1e-1]], np.float32)
        k.errorCovPost        = np.eye(2, np.float32)
        k.statePost           = np.zeros((2, 1), np.float32)
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
        if prev is None: return float(new)
        if abs(new - prev) > max_jump:
            return prev + np.sign(new - prev) * max_jump
        return new

    @staticmethod
    def _outer_contour_from_mask(bin_mask, eps_ratio=0.015):
        m = cv2.medianBlur(bin_mask, 3)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
        return approx.reshape(-1, 2)  # (N,2) int

    @staticmethod
    def _left_x_at_y_from_contour(contour_xy, y):
        xs = []
        Y = float(y)
        pts = contour_xy
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if (y1 <= Y < y2) or (y2 <= Y < y1):
                if y2 != y1:
                    t = (Y - y1) / (y2 - y1)
                    x = x1 + t * (x2 - x1)
                    xs.append(x)
        if not xs:
            return None
        return int(np.floor(min(xs)))

    # -------------- main API --------------
    def measure(self, conf_thresh=0.3):
        grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            return None, None

        img = self.converter.Convert(grab)
        frame = img.GetArray()
        H, W = frame.shape[:2]

        res = self.model(frame, verbose=False, conf=conf_thresh, imgsz=960)[0]
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

        # Prefer contour intersection for xL (smoother edge); fallback to scanline search
        xL_raw = None
        contour_xy = self._outer_contour_from_mask(groove_mask)
        if contour_xy is not None:
            xL_raw = self._left_x_at_y_from_contour(contour_xy, cy)
        if xL_raw is None:
            xL_raw, _, _ = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL_raw is None:
            return None, frame

        # Smooth mm/px
        mm_per_px = mm_per_px_raw if self.prev_mm_per_px is None else \
                    self._smooth(self.prev_mm_per_px, mm_per_px_raw, SMOOTH_BETA_SCALE)

        # First frame: seed and return raw distance
        if self.prev_cx is None:
            self.prev_cx, self.prev_xL, self.prev_mm_per_px = float(cx_raw), float(xL_raw), float(mm_per_px)
            self.cx_buf.clear(); self.cx_buf.append(float(cx_raw))
            dist0 = abs(cx_raw - xL_raw) * mm_per_px
            # optional draw
            vis = frame.copy()
            if self.draw_masks:
                vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)
            if self.draw_distance:
                cv2.circle(vis, (int(cx_raw), int(cy)), 5, (0, 0, 255), -1)
                cv2.circle(vis, (int(xL_raw), int(cy)), 5, (255, 0, 0), -1)
                cv2.line(vis, (int(xL_raw), int(cy)), (int(cx_raw), int(cy)), (0, 255, 0), 2)
            return dist0, vis

        # Convert mm caps into pixels
        cap_px    = max(1, int(round(MAX_JUMP_MM    / max(mm_per_px, 1e-6))))
        cap_cx_px = max(1, int(round(MAX_JUMP_CX_MM / max(mm_per_px, 1e-6))))

        # ----- cx: Kalman -> clamp -> debounce(1px) -> EWMA -> median -----
        _ = self.cx_kf.predict()
        meas = np.array([[np.float32(cx_raw)]])
        cx_corr = float(self.cx_kf.correct(meas)[0, 0])
        cx_corr = self._clamp_jump(self.prev_cx, cx_corr, cap_cx_px)
        cx_corr = self.cx_db.step(self.prev_cx, cx_corr)
        cx_s    = self._smooth(self.prev_cx, cx_corr, SMOOTH_BETA_CX)
        self.cx_buf.append(cx_s)
        cx = float(np.median(self.cx_buf))

        # ----- xL: clamp -> debounce(1px) -> EWMA -----
        xL_c = self._clamp_jump(self.prev_xL, xL_raw, cap_px)
        xL_c = self.xL_db.step(self.prev_xL, xL_c)
        xL   = self._smooth(self.prev_xL, xL_c, SMOOTH_BETA_X)

        distance_mm = abs(cx - xL) * mm_per_px

        # ----- draw -----
        vis = frame.copy()
        if self.draw_masks:
            vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)
        if self.draw_distance:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
            cv2.circle(vis, (int(round(xL)), int(round(cy))), 5, (255, 0, 0), -1)
            cv2.line(vis, (int(round(xL)), int(round(cy))), (int(round(cx)), int(round(cy))), (0, 255, 0), 2)
            cv2.putText(vis, f"{distance_mm:.2f} mm",
                        ((int(round(xL)) + int(round(cx))) // 2, max(int(round(cy)) - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # update state
        self.prev_cx, self.prev_xL, self.prev_mm_per_px = cx, xL, float(mm_per_px)
        return float(distance_mm), vis

    # optional helper for raw display
    def show_one_frame(self):
        grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            return None
        img = self.converter.Convert(grab)
        return img.GetArray()

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()
