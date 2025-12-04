# To check with videos, define folder to videos directory in:
# videos_dir  = os.path.join(project_dir, "data", "Curve_250808")
# folder "runs" with all structure should be in same path as script, so yolo could load pretrained weights
# press Space to pause/resume, Esc or q to quit

import os, time, json, cv2, numpy as np
from ultralytics import YOLO
from pathlib import Path

DRAW_MASK_AND_BBOX = True
DRAW_DISTANCE_TEXT = True
USE_GROOVE_BBOX_FOR_EDGES = False
SCALE_MM_PER_PX_MANUAL = None
ELECTRODE_DIAMETER_MM  = 4.3
RECORD_EVERY_MS = 500
OUTPUT_JSON_PATH = "measurements.json"

CLASS_GROOVE = 0
CLASS_WROD   = 1

CONF = 0.5
CONF_MARGIN = 0.8
SMOOTH_BETA_X = 0.9
SMOOTH_BETA_SCALE = 0.85
MAX_JUMP_PX = 2
MAX_JUMP_MM = 0.1

def bbox_from_mask(bin_mask):
    if bin_mask is None or bin_mask.sum() == 0: return None
    ys, xs = np.where(bin_mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def edge_xs_at_y_with_fallback(bin_mask, y, search=6):
    H, W = bin_mask.shape
    for dy in range(0, search+1):
        for yy in [y - dy, y + dy] if dy > 0 else [y]:
            if 0 <= yy < H:
                xs = np.where(bin_mask[yy] > 0)[0]
                if len(xs) > 0:
                    return int(xs.min()), int(xs.max()), int(yy)
    return None, None, None

def groove_edges_at_y(cy, groove_mask, groove_bbox, use_bbox, search=6):
    if use_bbox and groove_bbox is not None:
        gx1, gy1, gx2, gy2 = groove_bbox
        y_used = int(np.clip(cy, gy1, gy2))
        return int(gx1), int(gx2), y_used
    return edge_xs_at_y_with_fallback(groove_mask, cy, search=search)

def resize_masks_to_frame(res, H, W):
    if res.masks is None: return []
    raw = res.masks.data.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy().astype(float)
    out = []
    for m, cid, s in zip(raw, cls_ids, confs):
        m8 = (m * 255).astype(np.uint8)
        m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append((cid, m8, s))
    return out

def best_electrode_bbox(res):
    if res.boxes is None: return None
    bxyxy = res.boxes.xyxy.cpu().numpy()
    bcls  = res.boxes.cls.cpu().numpy().astype(int)
    bconf = res.boxes.conf.cpu().numpy()
    best = None
    for box, cid, conf in zip(bxyxy, bcls, bconf):
        if cid == CLASS_WROD:
            if (best is None) or (conf > best[-1]):
                best = (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf))
    return best

def iou_mask(a, b):
    if a is None or b is None: return 0.0
    ai, bi = (a > 0), (b > 0)
    inter = np.logical_and(ai, bi).sum()
    union = np.logical_or(ai, bi).sum()
    return 0.0 if union == 0 else float(inter) / float(union)

def smooth(prev, new, beta):
    return beta * prev + (1 - beta) * new

def clamp_jump(prev, new, max_jump):
    if abs(new - prev) > max_jump:
        return prev + np.sign(new - prev) * max_jump
    return new

def predict_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = Path.cwd().parents[2] / "data"
    # videos_dir  = os.path.join(data_dir, "Curve_250808")
    videos_dir  = "D:\ML_DL_AI_stuff\!!DoosanWelding2025\code\AtuomaticWeldingPositioning\DS_collection\original_video"

    vids = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    if not vids: print("No videos found!"); return
    for i, v in enumerate(vids): print(f"{i}: {v}")
    idx = int(input("Enter video index: "))
    video_path = os.path.join(videos_dir, vids[idx])

    # model_path = os.path.join(script_dir, "runs", "segment", "weld_seg_1031_1-", "weights", "best.pt")
    model_path = os.path.join(script_dir, "runs", "segment", "welding_seg_1203-", "weights", "best.pt")
    if not os.path.exists(model_path): print("Trained model not found!"); return
    model = YOLO(model_path)
    _ = model(np.zeros((512,512,3), dtype=np.uint8), verbose=False)

    cap = cv2.VideoCapture(video_path)
    paused = False
    fps_list = []
    record = {}
    next_record_ms = 0.0

    prev_groove_mask = None
    prev_cx = None
    prev_xL = None
    prev_mm_per_px = None

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        elif key == ord(' '): paused = not paused
        if paused: continue

        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        t0 = time.time()
        res = model(frame, verbose=False, conf=CONF)[0]
        fps = 1.0 / max(1e-6, time.time() - t0)
        fps_list.append(fps)

        vis = frame.copy()
        if DRAW_MASK_AND_BBOX:
            vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)

        masks = resize_masks_to_frame(res, H, W)
        groove_mc = [(m, s) for (cid, m, s) in masks if cid == CLASS_GROOVE]

        if groove_mc:
            max_conf = max(s for (_, s) in groove_mc)
            cand_idx = [i for i, (_, s) in enumerate(groove_mc) if (max_conf - s) <= CONF_MARGIN]
            if prev_groove_mask is None or len(cand_idx) == 1:
                groove_mask = (groove_mc[int(np.argmax([s for (_, s) in groove_mc]))][0] > 0).astype(np.uint8)
            else:
                best_i, best_iou = cand_idx[0], -1.0
                for i in cand_idx:
                    m = groove_mc[i][0]
                    iv = iou_mask(prev_groove_mask, m)
                    if iv > best_iou: best_i, best_iou = i, iv
                groove_mask = (groove_mc[best_i][0] > 0).astype(np.uint8)
        else:
            groove_mask = None

        groove_bbox = bbox_from_mask(groove_mask) if groove_mask is not None else None
        ebox = best_electrode_bbox(res)
        electrode_center = None
        mm_per_px = SCALE_MM_PER_PX_MANUAL

        if ebox is not None:
            x1,y1,x2,y2,_ = ebox
            cx_raw = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            electrode_center = (cx_raw, cy)
            width_px = max(1, x2 - x1)
            if mm_per_px is None:
                mm_per_px_raw = ELECTRODE_DIAMETER_MM / float(width_px)
            else:
                mm_per_px_raw = mm_per_px
        else:
            mm_per_px_raw = None

        if (groove_mask is not None) and (electrode_center is not None):
            cx_raw, cy = electrode_center
            xL_raw, xR_raw, y_used = groove_edges_at_y(cy, groove_mask, groove_bbox, USE_GROOVE_BBOX_FOR_EDGES, search=6)
            if xL_raw is not None:
                if prev_mm_per_px is None or mm_per_px_raw is None:
                    mm_per_px = mm_per_px_raw if mm_per_px_raw is not None else prev_mm_per_px
                else:
                    mm_per_px = smooth(prev_mm_per_px, mm_per_px_raw, SMOOTH_BETA_SCALE)

                if prev_cx is None:
                    cx = cx_raw
                else:
                    cap_px = max(MAX_JUMP_PX, int(round(MAX_JUMP_MM / max(mm_per_px or 1e-6, 1e-6))))
                    cx_c = clamp_jump(prev_cx, cx_raw, cap_px)
                    cx = smooth(prev_cx, cx_c, SMOOTH_BETA_X)

                if prev_xL is None:
                    xL = xL_raw
                else:
                    cap_px = max(MAX_JUMP_PX, int(round(MAX_JUMP_MM / max(mm_per_px or 1e-6, 1e-6))))
                    xL_c = clamp_jump(prev_xL, xL_raw, cap_px)
                    xL = smooth(prev_xL, xL_c, SMOOTH_BETA_X)

                if DRAW_DISTANCE_TEXT:
                    cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
                    cv2.circle(vis, (int(round(xL)), int(round(cy))), 5, (255, 0, 0), -1)
                    cv2.line(vis, (int(round(xL)), int(round(cy))), (int(round(cx)), int(round(cy))), (0, 255, 0), 2)

                distance_mm = None
                if mm_per_px is not None:
                    distance_mm = abs(cx - xL) * mm_per_px
                    cv2.putText(vis, f"{distance_mm:.2f} mm", ((int(round(xL)) + int(round(cx))) // 2, max(int(round(cy)) - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                prev_groove_mask = groove_mask
                prev_cx = float(cx)
                prev_xL = float(xL)
                if mm_per_px is not None: prev_mm_per_px = float(mm_per_px)
                groove_left_x = int(round(xL))
                groove_right_x = int(round(xR_raw)) if xR_raw is not None else None
                center_to_left_mm = abs(cx - xL) * mm_per_px if mm_per_px is not None else None
                center_to_right_mm = abs((xR_raw - cx)) * mm_per_px if (mm_per_px is not None and xR_raw is not None) else None
                r_px = (ELECTRODE_DIAMETER_MM / (mm_per_px or 1e-6)) / 2.0 if mm_per_px is not None else None
                clearance_left_mm  = max(0.0, (cx - (r_px or 0) - xL) * mm_per_px) if mm_per_px is not None else None
                clearance_right_mm = max(0.0, ((xR_raw or cx) - (cx + (r_px or 0))) * mm_per_px) if (mm_per_px is not None and xR_raw is not None) else None
            else:
                groove_left_x = groove_right_x = center_to_left_mm = center_to_right_mm = clearance_left_mm = clearance_right_mm = None
        else:
            groove_left_x = groove_right_x = center_to_left_mm = center_to_right_mm = clearance_left_mm = clearance_right_mm = None
            cx = cy = distance_mm = None

        cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.imshow("YOLO Real-time Prediction", vis)

        cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if cur_ms >= next_record_ms:
            record.update({
                "timestamp_ms": float(cur_ms),
                "mm_per_px": float(prev_mm_per_px) if prev_mm_per_px is not None else None,
                "electrode_center_px": [int(round(prev_cx)), int(round(cy))] if (prev_cx is not None and cy is not None) else None,
                "electrode_bbox_px": [int(ebox[0]), int(ebox[1]), int(ebox[2]), int(ebox[3])] if ebox else None,
                "electrode_diameter_mm": ELECTRODE_DIAMETER_MM,
                "groove_left_x_px_at_cy": int(groove_left_x) if groove_left_x is not None else None,
                "groove_right_x_px_at_cy": int(groove_right_x) if groove_right_x is not None else None,
                "center_to_left_mm": float(center_to_left_mm) if center_to_left_mm is not None else None,
                "center_to_right_mm": float(center_to_right_mm) if center_to_right_mm is not None else None,
                "clearance_left_mm": float(clearance_left_mm) if clearance_left_mm is not None else None,
                "clearance_right_mm": float(clearance_right_mm) if clearance_right_mm is not None else None,
            })
            try:
                with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
            except Exception as e:
                print("Failed to write JSON:", e)
            while next_record_ms <= cur_ms:
                next_record_ms += RECORD_EVERY_MS

    cap.release()
    cv2.destroyAllWindows()
    if fps_list: print(f"Average FPS: {sum(fps_list)/len(fps_list):.2f}")

if __name__ == "__main__":
    predict_video()
