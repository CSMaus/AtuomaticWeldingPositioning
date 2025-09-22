import cv2, numpy as np
from pypylon import pylon
from ultralytics import YOLO

CLASS_GROOVE = 0
CLASS_WROD = 1

class BaslerYoloMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.3, scale_mm_per_px=None):
        self.model = YOLO(weights_path)
        tlf = pylon.TlFactory.GetInstance()
        self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        dummy = np.zeros((512,512,3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)  # warm-up

    def _resize_masks(self, res, H, W):
        if res.masks is None: return []
        raw = res.masks.data.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        out = []
        for m, c in zip(raw, cls_ids):
            m8 = (m * 255).astype(np.uint8)
            m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
            out.append((c, m8))
        return out

    def _pick_largest(self, masks):
        if not masks: return None
        areas = [int((m > 0).sum()) for m in masks]
        return (masks[int(np.argmax(areas))] > 0).astype(np.uint8)

    def _best_wrod_bbox(self, res):
        if res.boxes is None: return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_ = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        best = None
        for box, c, s in zip(xyxy, cls_, conf):
            if c == CLASS_WROD and (best is None or s > best[-1]):
                best = (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(s))
        return best

    def _edge_xs_at_y(self, mask, y, search=6):
        H, W = mask.shape
        for dy in range(0, search+1):
            for yy in [y] if dy==0 else [y-dy, y+dy]:
                if 0 <= yy < H:
                    xs = np.where(mask[yy] > 0)[0]
                    if len(xs) > 0:
                        return int(xs.min()), int(xs.max()), int(yy)
        return None, None, None

    def measure(self):
        grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded(): return None
        img = self.converter.Convert(grab)
        frame = img.GetArray()
        H, W = frame.shape[:2]
        res = self.model(frame, verbose=False)[0]
        masks = self._resize_masks(res, H, W)
        groove_masks = [m for cid, m in masks if cid == CLASS_GROOVE]
        groove_mask = self._pick_largest(groove_masks)
        ebox = self._best_wrod_bbox(res)
        if groove_mask is None or ebox is None: return None
        x1, y1, x2, y2, _ = ebox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        width_px = max(1, x2 - x1)
        mm_per_px = self.scale_mm_per_px_manual or (self.electrode_diameter_mm / float(width_px))
        xL, xR, yy = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL is None: return None
        return abs(cx - xL) * mm_per_px

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()
