import cv2, numpy as np, random, os, csv, time
from ultralytics import YOLO
import matplotlib.pyplot as plt

CLASS_GROOVE = 0
CLASS_WROD = 1

VIDEO_PATH = r"D:\ML_DL_AI_stuff\!!DoosanWelding2025\data\Curve_250808\all move.mp4"
# WEIGHTS_PATH = r"runs/segment/weld_seg_0911_1-/weights/best.pt"
WEIGHTS_PATH = r"runs/segment/weld_seg_1103-2/weights/best.pt"
CONF = 0.3
STATIC_RUNS = 50
ELECTRODE_MM = 4.3
SCALE_MM_PER_PX = None

class YoloFrameMeasurer:
    def __init__(self, weights_path, electrode_diameter_mm=4.3, scale_mm_per_px=None, draw_masks=True, draw_distance=True):
        self.model = YOLO(weights_path)
        self.electrode_diameter_mm = float(electrode_diameter_mm)
        self.scale_mm_per_px_manual = scale_mm_per_px
        self.draw_masks = draw_masks
        self.draw_distance = draw_distance
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
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
    def measure_on_frame(self, frame, conf_thresh=0.3):
        H, W = frame.shape[:2]
        res = self.model(frame, verbose=False, conf=conf_thresh)[0]
        masks = self._resize_masks(res, H, W)
        groove_masks = [m for cid, m in masks if cid == CLASS_GROOVE]
        groove_mask = self._pick_largest(groove_masks)
        ebox = self._best_wrod_bbox(res)
        if groove_mask is None or ebox is None:
            return False, {}, frame
        x1, y1, x2, y2, _ = ebox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        width_px = max(1, x2 - x1)
        mm_per_px = self.scale_mm_per_px_manual or (self.electrode_diameter_mm / float(width_px))
        xL, xR, yy = self._edge_xs_at_y(groove_mask, cy, search=6)
        if xL is None:
            return False, {}, frame
        distance_mm = abs(cx - xL) * mm_per_px
        vis = frame.copy()
        vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(vis, (xL, cy), 5, (255, 0, 0), -1)
        cv2.line(vis, (xL, cy), (cx, cy), (0, 255, 0), 2)
        cv2.putText(vis, f"{distance_mm:.2f} mm", ((xL + cx) // 2, max(cy - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        metrics = dict(xL_px=int(xL), cx_px=int(cx), cy_px=int(cy), mm_per_px=float(mm_per_px), dist_mm=float(distance_mm))
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


