import cv2, numpy as np
from pypylon import pylon
from ultralytics import YOLO

CLASS_GROOVE = 0
CLASS_WROD = 1

# new params for smoothing left groove edge detection (to remove fluctuation and spikes)
SMOOTH_BETA_X = 0.85  # parameter used for smoothing predictions. read "_clamp_smooth" static method
MAX_JUMP_MM = 0.1  # not used
MIN_JUMP_MM = 0.31  # if the difference beveen predictions in current and previous frames less, do smoothing


class BaslerYoloMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.5, scale_mm_per_px=None, draw_masks=True, draw_distance=True):
        self.model = YOLO(weights_path)

        # define fields to connect camera properly
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(devices[0]))
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # define parameters for NN prediction and detections
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        self.draw_masks = draw_masks
        self.draw_distance = draw_distance

        # warm-up the model so it will not freeze in the beginning
        dummy = np.zeros((512,512,3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)  # warm-up

        self.prev_xL = None


    def _resize_masks(self, res, H, W):
        """ resize mask from YOLO input image size to the original size from camera"""
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
        """ sort all predictions to take best"""
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

    def resize_to_32_multiple(self, image):
        # not 32 multiple, but 16 with shrinking image size twice. it looks like required by MaskRCNN
        h, w = image.shape[:2]
        new_h = (h // 32) * 16
        new_w = (w // 32) * 16
        return cv2.resize(image, (new_w, new_h))
    
    @staticmethod
    def _smooth(prev, new, beta):
        return int(beta * prev + (1 - beta) * new)

    @staticmethod
    def _clamp_jump(prev, new, max_jump):
        if abs(new - prev) > max_jump:
            return prev + np.sign(new - prev) * max_jump
        return new
    
    @staticmethod
    def _clamp_smooth(prev, new, min_jump_ignore, beta):
        """ apply smooting of predicted left groove edge value
        by using previous prediction
        but we apply smoothing only in case, if the difference between predicted groove edge value in
        current frame and previous frame is less than min_jump_ignore """
        if abs(new - prev) < min_jump_ignore:
            return int(beta * prev + (1 - beta) * new)
            # return prev + np.sign(new - prev) * max_jump
        else:
            return int((1 - beta) * prev + beta * new)
        # return new

    def measure(self, conf_thresh = 0.3):
        """
        comment or delete "cv2.line(...)" to not draw distance line
        lines with "cv2.circle(...)" draw dots of the W-rod and groove edge. between dots we calculate distance
        :param conf_thresh: confidence threshold. All predictions below it will not be displayed
        :return:
        """

        # _____________________ get image (frame) from camera
        grabResult = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grabResult.GrabSucceeded():
            return 100, None
        image = self.converter.Convert(grabResult)
        frame = image.GetArray()
        # frame = self.resize_to_32_multiple(frame)  # no need for this
        H, W = frame.shape[:2]
        # _____________________

        # ____________ make predictions and get results with confidence more than threshold (default 0.5)
        res = self.model(frame, verbose=False, conf=conf_thresh)[0]
        masks = self._resize_masks(res, H, W)
        groove_masks = [m for cid, m in masks if cid == CLASS_GROOVE]
        groove_mask = self._pick_largest(groove_masks)
        ebox = self._best_wrod_bbox(res)
        if groove_mask is None or ebox is None:
            # cv2.imshow('Image Test', frame)
            # cv2.waitKey(0)
            return 100, frame

        # _____________________  using known width of W-Rod in mm (self.electrode_diameter_mm) we define:
        # how many mm is in one pixel, so we could calculate distance between W-Rod and left groove edge
        x1, y1, x2, y2, _ = ebox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        width_px = max(1, x2 - x1)
        mm_per_px = self.scale_mm_per_px_manual or (self.electrode_diameter_mm / float(width_px))
        xL_raw, _, _ = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL_raw is None:
            return 100, frame


        # new code for smoothing left groove edge detection
        # to turn it off, comment out few lines below
        # and change xL_raw definistion above to be xL: xL_raw -> xL
        if self.prev_xL is None:
            self.prev_xL = xL_raw
        cap_px = max(1, int(round(MAX_JUMP_MM / max(mm_per_px, 1e-6))))
        # xL_c = self._clamp_jump(self.prev_xL, xL_raw, cap_px)
        # xL = self._smooth(self.prev_xL, xL_raw, SMOOTH_BETA_X)
        xL = self._clamp_smooth(self.prev_xL, xL_raw, MIN_JUMP_MM, SMOOTH_BETA_X)

        distance_mm = abs(cx - xL) * mm_per_px

        # ____________________ draw masks and line of measured distance in  frame
        frame = cv2.addWeighted(frame, 0.6, res.plot(), 0.4, 0)
        print("xL: ", xL)
        print("xL vaariable type: ", type(xL))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (xL, cy), 5, (255, 0, 0), -1)
        cv2.line(frame, (xL, cy), (cx, cy), (0, 255, 0), 2)
        # cv2.putText(frame, f"{distance_mm:.2f} mm", ((xL + cx) // 2, cy - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return distance_mm, frame

    def show_one_frame(self):
        # in case if need to grab and display image without making any predictions
        # this function tested to check clear camera speed (FPS) without NN
        grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            return None
        img = self.converter.Convert(grab)
        frame = img.GetArray()
        # frame = self.resize_to_32_multiple(frame)  # no need for this
        return frame

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()
