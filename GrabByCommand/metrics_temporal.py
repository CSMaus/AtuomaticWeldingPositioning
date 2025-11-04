import cv2, numpy as np, random, os, csv, time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from collections import deque

CLASS_GROOVE = 0
CLASS_WROD = 1

VIDEO_PATH = r"D:\ML_DL_AI_stuff\!!DoosanWelding2025\data\Curve_250808\all move.mp4"
# WEIGHTS_PATH = r"runs/segment/weld_seg_0911_1-/weights/best.pt"
WEIGHTS_PATH = r"runs/segment/weld_seg_1103-2/weights/best.pt"
CONF = 0.3
STATIC_RUNS = 5
ELECTRODE_MM = 4.3
SCALE_MM_PER_PX = None
CONF_MARGIN = 0.2
SMOOTH_BETA_X = 0.9
SMOOTH_BETA_SCALE = 0.85
MAX_JUMP_PX = 2
MAX_JUMP_MM = 0.1

SMOOTH_BETA_CX = 0.93
MAX_JUMP_CX_MM = 0.08
MEDIAN_CX_WIN  = 5
LAMBDA_IOU_BOX = 0.6

fps_list = []

class YoloFrameMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.3, scale_mm_per_px=None, draw_masks=True, draw_distance=True):
        self.model = YOLO(weights_path)
        self.model.to('cuda')
        self.model.fuse()
        torch.backends.cudnn.benchmark = True  # it should be already in cuda, but anyway
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        self.draw_masks = draw_masks
        self.draw_distance = draw_distance
        self.prev_groove_mask = None
        self.conf_margin = CONF_MARGIN
        self.prev_xL = None
        self.prev_cx = None
        self.prev_mm_per_px = None
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)

        # this is new - recheck
        self.prev_ebox = None
        self.cx_kf = make_kalman_1d()
        self.cx_buf = deque(maxlen=MEDIAN_CX_WIN)
    def reset_state(self):
        self.prev_groove_mask = None
        self.prev_xL = None
        self.prev_cx = None
        self.prev_mm_per_px = None

        self.prev_ebox = None
        self.cx_buf.clear()
        self.cx_kf = make_kalman_1d()
    @staticmethod
    def _resize_masks(res, H, W):
        if res.masks is None: return []
        raw = res.masks.data.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy().astype(float)
        out = []
        for m, c, s in zip(raw, cls_ids, confs):
            m8 = (m * 255).astype(np.uint8)
            m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
            out.append((c, m8, s))
        return out
    # @staticmethod
    def _best_wrod_bbox(self, res):  # this is new - recheck
        if res.boxes is None: return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_ = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        cand = [(tuple(map(int, box[:4])), float(s)) for box, c, s in zip(xyxy, cls_, conf) if c == CLASS_WROD]
        if not cand: return None
        if self.prev_ebox is None:
            box, s = max(cand, key=lambda t: t[1]);
            return (*box, s)
        scored = []
        for box, s in cand:
            score = s + LAMBDA_IOU_BOX * iou_box((*box, 0.0), self.prev_ebox)
            scored.append((score, box, s))
        _, box, s = max(scored, key=lambda t: t[0])
        return (*box, s)
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
    def _iou(a, b):
        if a is None or b is None: return 0.0
        ai = (a > 0)
        bi = (b > 0)
        inter = np.logical_and(ai, bi).sum()
        union = np.logical_or(ai, bi).sum()
        if union == 0: return 0.0
        return float(inter) / float(union)
    def _select_groove_mask(self, groove_masks_conf):
        if not groove_masks_conf: return None
        confs = [s for (_, s) in groove_masks_conf]
        max_conf = max(confs)
        idxs_close = [i for i, s in enumerate(confs) if (max_conf - s) <= self.conf_margin]
        if len(idxs_close) == 1 or self.prev_groove_mask is None:
            best_i = int(np.argmax(confs))
            return (groove_masks_conf[best_i][0]).astype(np.uint8)
        best_iou = -1.0
        best_idx = idxs_close[0]
        for i in idxs_close:
            m = groove_masks_conf[i][0]
            iou = self._iou(self.prev_groove_mask, m)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return (groove_masks_conf[best_idx][0]).astype(np.uint8)
    @staticmethod
    def _smooth(prev, new, beta):
        return beta * prev + (1 - beta) * new
    @staticmethod
    def _clamp_jump(prev, new, max_jump):
        if abs(new - prev) > max_jump:
            return prev + np.sign(new - prev) * max_jump
        return new
    def measure_on_frame(self, frame, conf_thresh=0.3):
        H, W = frame.shape[:2]
        # res = self.model(frame, verbose=False, conf=conf_thresh)[0]
        # maybe it will improve accuracy, especially +-1 pixel fluctuations
        res = self.model(frame, verbose=False, conf=conf_thresh, imgsz=960, device=0, half=True,
                         classes=[CLASS_GROOVE, CLASS_WROD])[0]
        masks = self._resize_masks(res, H, W)
        groove_mc = [(m, s) for cid, m, s in masks if cid == CLASS_GROOVE]
        groove_mask = self._select_groove_mask(groove_mc)

        ebox = self._best_wrod_bbox(res) # this is new - recheck

        if groove_mask is None or ebox is None:
            return False, {}, frame
        # # this is new - recheck
        self.prev_ebox = ebox

        x1, y1, x2, y2, _ = ebox
        if x2 < x1: x1, x2 = x2, x1  # no bbox flip; todo: delete it later
        cx_raw, cy = (x1 + x2) // 2, (y1 + y2) // 2
        width_px = max(1, x2 - x1)
        mm_per_px_raw = self.scale_mm_per_px_manual or (self.electrode_diameter_mm / float(width_px))
        xL_raw, xR, yy = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL_raw is None:
            return False, {}, frame
        if self.prev_mm_per_px is None:
            mm_per_px = mm_per_px_raw
        else:
            mm_per_px = self._smooth(self.prev_mm_per_px, mm_per_px_raw, SMOOTH_BETA_SCALE)
        if self.prev_cx is None:
            # ensure x1<=x2 just in case
            if x2 < x1: x1, x2 = x2, x1
            cx = float(cx_raw)
            xL = float(xL_raw)
            mm_per_px = float(mm_per_px_raw)

            # seed state
            self.cx_kf.statePost = np.array([[np.float32(cx)], [0.0]], dtype=np.float32)
            self.prev_cx = cx
            self.prev_xL = xL
            self.prev_mm_per_px = mm_per_px
            self.prev_ebox = (x1, y1, x2, y2, _)
            self.cx_buf.clear();
            self.cx_buf.append(cx)

            distance_mm = abs(cx - xL) * mm_per_px
            vis = frame.copy()
            if self.draw_distance:
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.circle(vis, (int(xL), int(cy)), 5, (255, 0, 0), -1)
                cv2.line(vis, (int(xL), int(cy)), (int(cx), int(cy)), (0, 255, 0), 2)
                cv2.putText(vis, f"{distance_mm:.2f} mm",
                            ((int(xL) + int(cx)) // 2, max(int(cy) - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            metrics = dict(xL_px=int(xL), cx_px=int(cx), cy_px=int(cy),
                           mm_per_px=float(mm_per_px), dist_mm=float(distance_mm))
            return True, metrics, vis

        cap_px = max(1, int(round(MAX_JUMP_MM / max(mm_per_px, 1e-6))))
        '''if self.prev_cx is None:
            cx = cx_raw
        else:
            # cx_c = self._clamp_jump(self.prev_cx, cx_raw, MAX_JUMP_PX)
            cx_c = self._clamp_jump(self.prev_cx, cx_raw, cap_px)
            cx = self._smooth(self.prev_cx, cx_c, SMOOTH_BETA_X)'''

        # this is new - recheck kalman
        cx_pred = float(self.cx_kf.predict()[0, 0])
        meas = np.array([[np.float32(cx_raw)]])
        cx_corr = float(self.cx_kf.correct(meas)[0, 0])
        cap_cx_px = max(1, int(round(MAX_JUMP_CX_MM / max(mm_per_px, 1e-6))))
        if self.prev_cx is not None:
            cx_corr = self._clamp_jump(self.prev_cx, cx_corr, cap_cx_px)
        cx_s = self._smooth(self.prev_cx if self.prev_cx is not None else cx_corr, cx_corr, SMOOTH_BETA_CX)
        self.cx_buf.append(cx_s)
        cx = float(np.median(self.cx_buf))


        if self.prev_xL is None:
            xL = xL_raw
        else:
            # xL_c = self._clamp_jump(self.prev_xL, xL_raw, MAX_JUMP_PX)
            xL_c = self._clamp_jump(self.prev_xL, xL_raw, cap_px)
            xL = self._smooth(self.prev_xL, xL_c, SMOOTH_BETA_X)
        distance_mm = abs(cx - xL) * mm_per_px
        vis = frame.copy()
        # if self.draw_masks:  # do not draw masks, maybe it will speed up
        #     vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)
        if self.draw_distance:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
            cv2.circle(vis, (int(round(xL)), int(round(cy))), 5, (255, 0, 0), -1)
            cv2.line(vis, (int(round(xL)), int(round(cy))), (int(round(cx)), int(round(cy))), (0, 255, 0), 2)
            cv2.putText(vis, f"{distance_mm:.2f} mm", ((int(round(xL)) + int(round(cx))) // 2, max(int(round(cy)) - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        self.prev_groove_mask = groove_mask
        self.prev_cx = float(cx)
        self.prev_xL = float(xL)
        self.prev_mm_per_px = float(mm_per_px)
        metrics = dict(xL_px=int(round(xL)), cx_px=int(round(cx)), cy_px=int(cy), mm_per_px=float(mm_per_px), dist_mm=float(distance_mm))
        return True, metrics, vis

# this is new - recheck
def make_kalman_1d():
    k = cv2.KalmanFilter(2,1)
    k.transitionMatrix   = np.array([[1,1],[0,1]], np.float32)
    k.measurementMatrix  = np.array([[1,0]], np.float32)
    k.processNoiseCov    = np.eye(2, dtype=np.float32) * 1e-2
    k.measurementNoiseCov= np.array([[1e-1]], np.float32)
    k.errorCovPost       = np.eye(2, dtype=np.float32)
    k.statePost          = np.zeros((2,1), np.float32)
    return k

# this is new - recheck
def iou_box(a, b):
    if a is None or b is None: return 0.0
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    aarea = max(0, ax2-ax1)*max(0, ay2-ay1)
    barea = max(0, bx2-bx1)*max(0, by2-by1)
    den = aarea + barea - inter
    return 0.0 if den<=0 else inter/den

def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])

def plot_series(x, y, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def summarize(name, arr):
    if len(arr) == 0:
        return f"{name}: no data"
    a = np.array(arr, dtype=float)
    return f"{name}: n={len(a)}, mean={a.mean():.4f}, std={a.std(ddof=1):.4f}, min={a.min():.4f}, max={a.max():.4f}"

def run_static_test(cap, measurer, conf, runs=50):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        target_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    else:
        target_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            return {"frame_index": None, "rows": [], "summary": "Static test: failed to read frame."}
    measurer.reset_state()
    rows = []
    for i in range(runs):
        ok, m, _ = measurer.measure_on_frame(frame, conf_thresh=conf)
        if ok:
            rows.append(dict(run=i, frame_index=target_idx, **m))
        else:
            rows.append(dict(run=i, frame_index=target_idx, xL_px="", cx_px="", cy_px="", mm_per_px="", dist_mm=""))
    xL_list = [r["xL_px"] for r in rows if r["xL_px"] != ""]
    cx_list = [r["cx_px"] for r in rows if r["cx_px"] != ""]
    d_list  = [r["dist_mm"] for r in rows if r["dist_mm"] != ""]
    summary = "\n".join([
        f"Static test at frame={target_idx}",
        summarize("xL_px", xL_list),
        summarize("cx_px", cx_list),
        summarize("dist_mm", d_list),
    ])
    return {"frame_index": target_idx, "rows": rows, "summary": summary}

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    measurer = YoloFrameMeasurer(WEIGHTS_PATH, electrode_diameter_mm=ELECTRODE_MM, scale_mm_per_px=SCALE_MM_PER_PX, draw_masks=True, draw_distance=True)
    static_result = run_static_test(cap, measurer, CONF, runs=STATIC_RUNS)
    static_rows = static_result["rows"]
    print(static_result["summary"])
    if len(static_rows) > 0:
        write_csv("metrics_static.csv", static_rows, header=["run","frame_index","xL_px","cx_px","cy_px","mm_per_px","dist_mm"])
        runs_idx = [r["run"] for r in static_rows if r["xL_px"] != ""]
        xL_vals  = [r["xL_px"] for r in static_rows if r["xL_px"] != ""]
        cx_vals  = [r["cx_px"] for r in static_rows if r["cx_px"] != ""]
        d_vals   = [r["dist_mm"] for r in static_rows if r["dist_mm"] != ""]
        if len(xL_vals) > 0: plot_series(runs_idx, xL_vals, "Static: Groove left-edge x", "run", "xL_px", "static_xL_px.png")
        if len(cx_vals) > 0: plot_series(runs_idx, cx_vals, "Static: Electrode center x", "run", "cx_px", "static_cx_px.png")
        if len(d_vals) > 0: plot_series(runs_idx, d_vals, "Static: Distance", "run", "dist_mm", "static_dist_mm.png")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    measurer.reset_state()
    dyn_rows = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    start_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()

        okm, m, vis = measurer.measure_on_frame(frame, conf_thresh=CONF)
        if okm:
            txt = f"d={m['dist_mm']:.2f} mm | xL={m['xL_px']} | cx={m['cx_px']}"
        else:
            txt = "NO MEASURE"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        fps_m = 1.0 / max(1e-6, time.time() - t0)
        cv2.putText(vis, str(fps_m), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,55,55), 2, cv2.LINE_AA)
        cv2.imshow("YOLO Video Measure", vis)
        fps_list.append(fps_m)
        # time.sleep(0.05)  # it slowed down the whole predictions - bcs I wanted to check without kalman and others
        # need to check fps
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t_sec = frame_idx / fps if fps > 0 else (time.time() - start_t)
        row = dict(frame_index=frame_idx, t_sec=round(t_sec, 4))
        if okm: row.update(m)
        dyn_rows.append(row)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(dyn_rows) > 0:
        header = ["frame_index","t_sec","xL_px","cx_px","cy_px","mm_per_px","dist_mm"]
        write_csv("metrics_dynamic.csv", dyn_rows, header=header)
        t_vals  = [r["t_sec"] for r in dyn_rows if r.get("xL_px","") != ""]
        xL_vals = [r["xL_px"] for r in dyn_rows if r.get("xL_px","") != ""]
        cx_vals = [r["cx_px"] for r in dyn_rows if r.get("cx_px","") != ""]
        d_vals  = [r["dist_mm"] for r in dyn_rows if r.get("dist_mm","") != ""]
        if len(xL_vals) > 0: plot_series(t_vals, xL_vals, "Video: Groove left-edge x over time", "time (s)", "xL_px", "video_xL_px.png")
        if len(cx_vals) > 0: plot_series(t_vals, cx_vals, "Video: Electrode center x over time", "time (s)", "cx_px", "video_cx_px.png")
        if len(d_vals) > 0: plot_series(t_vals, d_vals, "Video: Distance over time", "time (s)", "dist_mm", "video_dist_mm.png")
        if len(xL_vals) > 1: print(summarize("xL_px", xL_vals))
        if len(cx_vals) > 1: print(summarize("cx_px", cx_vals))
        if len(d_vals)  > 1: print(summarize("dist_mm", d_vals))
        print("Saved: metrics_dynamic.csv, video_xL_px.png, video_cx_px.png, video_dist_mm.png")
    if len(static_rows) > 0:
        print("Saved: metrics_static.csv, static_xL_px.png, static_cx_px.png, static_dist_mm.png")
    if fps_list: print(f"Average FPS: {sum(fps_list) / len(fps_list):.2f}")

if __name__ == "__main__":
    main()
