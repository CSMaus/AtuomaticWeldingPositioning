import cv2, numpy as np, random, os, csv, time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque

CLASS_GROOVE = 0
CLASS_WROD = 1

VIDEO_PATH = r"D:\ML_DL_AI_stuff\!!DoosanWelding2025\data\Curve_250808\all move.mp4"
WEIGHTS_PATH = r"runs/segment/weld_seg_0911_1-/weights/best.pt"
CONF = 0.3
STATIC_RUNS = 50
ELECTRODE_MM = 4.3
SCALE_MM_PER_PX = None

CONF_MARGIN = 0.18
SMOOTH_BETA_X = 0.9
SMOOTH_BETA_SCALE = 0.85
MAX_JUMP_MM = 0.3
MEDIAN_LEN = 5

ALPHA_CONF = 0.3
WIDTH_TOL = 0.12

KALMAN_Q = 0.01
KALMAN_R = 1.0

class YoloFrameMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.3, scale_mm_per_px=None, draw_masks=True, draw_distance=True):
        self.model = YOLO(weights_path)
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        self.draw_masks = draw_masks
        self.draw_distance = draw_distance
        self.prev_groove_mask = None
        self.prev_bbox = None
        self.prev_xL = None
        self.prev_cx = None
        self.prev_mm_per_px = None
        self.conf_margin = CONF_MARGIN
        self.buf_cx = deque(maxlen=MEDIAN_LEN)
        self.buf_xL = deque(maxlen=MEDIAN_LEN)
        self.kx = None
        self.kp = None
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
    def reset_state(self):
        self.prev_groove_mask = None
        self.prev_bbox = None
        self.prev_xL = None
        self.prev_cx = None
        self.prev_mm_per_px = None
        self.buf_cx.clear()
        self.buf_xL.clear()
        self.kx = None
        self.kp = None
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
    def _iou_mask(a, b):
        if a is None or b is None: return 0.0
        ai = (a > 0)
        bi = (b > 0)
        inter = np.logical_and(ai, bi).sum()
        uni = np.logical_or(ai, bi).sum()
        if uni == 0: return 0.0
        return float(inter) / float(uni)
    @staticmethod
    def _iou_box(b1, b2):
        if b1 is None or b2 is None: return 0.0
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        xi1, yi1 = max(x11, x21), max(y11, y21)
        xi2, yi2 = min(x12, x22), min(y12, y22)
        iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
        inter = iw * ih
        a1 = max(0, x12 - x11) * max(0, y12 - y11)
        a2 = max(0, x22 - x21) * max(0, y22 - y21)
        uni = a1 + a2 - inter
        if uni <= 0: return 0.0
        return inter / uni
    def _select_groove_mask(self, groove_masks_conf):
        if not groove_masks_conf: return None
        confs = [s for (_, s) in groove_masks_conf]
        maxc = max(confs)
        idxs = [i for i, s in enumerate(confs) if (maxc - s) <= self.conf_margin]
        if len(idxs) == 1 or self.prev_groove_mask is None:
            i = int(np.argmax(confs))
            return (groove_masks_conf[i][0]).astype(np.uint8)
        best, bi = -1.0, idxs[0]
        for i in idxs:
            m = groove_masks_conf[i][0]
            v = self._iou_mask(self.prev_groove_mask, m)
            if v > best:
                best, bi = v, i
        return (groove_masks_conf[bi][0]).astype(np.uint8)
    def _select_wrod_bbox(self, res):
        if res.boxes is None: return None
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        cls_ = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy().astype(float)
        cand = [(tuple(map(int, xyxy[i])), conf[i]) for i in range(len(cls_)) if cls_[i] == CLASS_WROD]
        if not cand: return None
        if self.prev_bbox is None:
            j = int(np.argmax([c[1] for c in cand]))
            b = cand[j][0]
            return (b[0], b[1], b[2], b[3], float(cand[j][1]))
        pw = max(1, self.prev_bbox[2] - self.prev_bbox[0])
        best_s, best = -1e9, None
        for b, s in cand:
            iou = self._iou_box(self.prev_bbox, b)
            bw = max(1, b[2] - b[0])
            dw = abs(bw - pw) / float(pw)
            pen = max(0.0, (dw - WIDTH_TOL))
            score = (1 - ALPHA_CONF) * iou + ALPHA_CONF * s - pen
            if score > best_s:
                best_s, best = score, (b[0], b[1], b[2], b[3], float(s))
        return best
    @staticmethod
    def _smooth(prev, new, beta):
        return beta * prev + (1 - beta) * new
    @staticmethod
    def _clamp_jump(prev, new, cap):
        if abs(new - prev) > cap:
            return prev + np.sign(new - prev) * cap
        return new
    def _kfilter(self, z):
        if self.kx is None:
            self.kx = float(z)
            self.kp = 1.0
            return self.kx
        xp = self.kx
        pp = self.kp + KALMAN_Q
        k = pp / (pp + KALMAN_R)
        self.kx = xp + k * (z - xp)
        self.kp = (1 - k) * pp
        return self.kx
    def measure_on_frame(self, frame, conf_thresh=0.3):
        H, W = frame.shape[:2]
        res = self.model(frame, verbose=False, conf=conf_thresh)[0]
        masks = self._resize_masks(res, H, W)
        groove_mc = [(m, s) for cid, m, s in masks if cid == CLASS_GROOVE]
        groove_mask = self._select_groove_mask(groove_mc)
        ebox = self._select_wrod_bbox(res)
        if groove_mask is None or ebox is None:
            return False, {}, frame
        x1, y1, x2, y2, _ = ebox
        cx_raw, cy = (x1 + x2) // 2, (y1 + y2) // 2
        width_px = max(1, x2 - x1)
        mm_per_px_raw = self.scale_mm_per_px_manual or (self.electrode_diameter_mm / float(width_px))
        xL_raw, _, _ = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL_raw is None:
            return False, {}, frame
        if self.prev_mm_per_px is None:
            mm_per_px = mm_per_px_raw
        else:
            mm_per_px = self._smooth(self.prev_mm_per_px, mm_per_px_raw, SMOOTH_BETA_SCALE)
        cap_px = max(1, int(round(MAX_JUMP_MM / max(mm_per_px, 1e-9))))
        if self.prev_cx is None:
            cx_c = cx_raw
        else:
            cx_c = self._clamp_jump(self.prev_cx, cx_raw, cap_px)
        if self.prev_xL is None:
            xL_c = xL_raw
        else:
            xL_c = self._clamp_jump(self.prev_xL, xL_raw, cap_px)
        cx_f = self._kfilter(cx_c)
        if self.prev_cx is None:
            cx = cx_f
        else:
            cx = self._smooth(self.prev_cx, cx_f, SMOOTH_BETA_X)
        if self.prev_xL is None:
            xL = xL_c
        else:
            xL = self._smooth(self.prev_xL, xL_c, SMOOTH_BETA_X)
        self.buf_cx.append(cx)
        self.buf_xL.append(xL)
        cx = float(np.median(self.buf_cx))
        xL = float(np.median(self.buf_xL))
        distance_mm = abs(cx - xL) * mm_per_px
        vis = frame.copy()
        if self.draw_masks:
            vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)
        if self.draw_distance:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
            cv2.circle(vis, (int(round(xL)), int(round(cy))), 5, (255, 0, 0), -1)
            cv2.line(vis, (int(round(xL)), int(round(cy))), (int(round(cx)), int(round(cy))), (0, 255, 0), 2)
            cv2.putText(vis, f"{distance_mm:.2f} mm", ((int(round(xL)) + int(round(cx))) // 2, max(int(round(cy)) - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        self.prev_groove_mask = groove_mask
        self.prev_bbox = (x1, y1, x2, y2)
        self.prev_cx = float(cx)
        self.prev_xL = float(xL)
        self.prev_mm_per_px = float(mm_per_px)
        metrics = dict(xL_px=int(round(xL)), cx_px=int(round(cx)), cy_px=int(cy), mm_per_px=float(mm_per_px), dist_mm=float(distance_mm))
        return True, metrics, vis

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
        okm, m, vis = measurer.measure_on_frame(frame, conf_thresh=CONF)
        if okm:
            txt = f"d={m['dist_mm']:.2f} mm | xL={m['xL_px']} | cx={m['cx_px']}"
        else:
            txt = "NO MEASURE"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("YOLO Video Measure", vis)
        time.sleep(0.05)
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

if __name__ == "__main__":
    main()

