import os, time, json, cv2, numpy as np
from ultralytics import YOLO
from pypylon import pylon

DRAW_MASK_AND_BBOX    = True
DRAW_DISTANCE_TEXT    = True
USE_GROOVE_BBOX_FOR_EDGES = False
SCALE_MM_PER_PX_MANUAL = None    # e.g., 0.015 mm/px; None -> auto from bbox width
ELECTRODE_DIAMETER_MM  = 4.0  # (mm)
RECORD_EVERY_MS        = 500 # save cadence (ms)
OUTPUT_JSON_PATH       = "measurements.json"

CLASS_GROOVE = 0
CLASS_WROD   = 1

def bbox_from_mask(bin_mask):
    if bin_mask is None or bin_mask.sum() == 0:
        return None
    ys, xs = np.where(bin_mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)

def groove_edges_at_y(cy, groove_mask, groove_bbox, use_bbox, search=6):
    """
    Return left_x, right_x, y_used.
    - If use_bbox: use bbox vertical edges at y=clamped(cy).
    - Else: use segmentation mask row (with fallback +/- search).
    """
    if use_bbox and groove_bbox is not None:
        gx1, gy1, gx2, gy2 = groove_bbox
        y_used = int(np.clip(cy, gy1, gy2))
        return int(gx1), int(gx2), y_used
    return edge_xs_at_y_with_fallback(groove_mask, cy, search=search)

def resize_masks_to_frame(res, H, W):
    """Return list of (cls_id, mask_uint8) resized to frame size."""
    if res.masks is None:
        return []
    raw_masks = res.masks.data.cpu().numpy()
    cls_ids   = res.boxes.cls.cpu().numpy().astype(int)
    out = []
    for m, cls_id in zip(raw_masks, cls_ids):
        m8 = (m * 255).astype(np.uint8)
        m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append((cls_id, m8))
    return out

def pick_largest_mask(masks_for_class):
    """Pick largest area mask (uint8 >0) from a list."""
    if not masks_for_class:
        return None
    areas = [int((m > 0).sum()) for m in masks_for_class]
    return (masks_for_class[int(np.argmax(areas))] > 0).astype(np.uint8)

def best_electrode_bbox(res):
    """Return best (x1,y1,x2,y2, conf) for class=CLASS_WROD."""
    if res.boxes is None:
        return None
    bxyxy = res.boxes.xyxy.cpu().numpy()
    bcls  = res.boxes.cls.cpu().numpy().astype(int)
    bconf = res.boxes.conf.cpu().numpy()
    best = None
    for box, cls_id, conf in zip(bxyxy, bcls, bconf):
        if cls_id == CLASS_WROD:
            if (best is None) or (conf > best[-1]):
                best = (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf))
    return best

def edge_xs_at_y_with_fallback(bin_mask, y, search=6):
    """
    Find groove left/right x at row y. If empty, search +/- dy (small window).
    Return (x_left, x_right, y_used) or (None, None, None).
    """
    H, W = bin_mask.shape
    for dy in range(0, search+1):
        for yy in [y - dy, y + dy] if dy > 0 else [y]:
            if 0 <= yy < H:
                xs = np.where(bin_mask[yy] > 0)[0]
                if len(xs) > 0:
                    return int(xs.min()), int(xs.max()), int(yy)
    return None, None, None

def draw_hline_with_text(img, y, x1, x2, txt=None, color=(0,255,0), thickness=2):
    cv2.line(img, (x1, y), (x2, y), color, thickness)
    if txt:
        midx = (x1 + x2) // 2
        cv2.putText(img, txt, (midx, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

def predict_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(script_dir, "runs", "segment", "weld_seg_0911_1-", "weights", "best.pt")
    if not os.path.exists(model_path):
        print("Trained model not found!"); return
    model = YOLO(model_path)

    # --- Basler Camera Setup ---
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.PixelFormat.SetValue("BayerBG8")
    camera.MaxNumBuffer = 3
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    fps_list = []

    record = {}
    next_record_ms = 0.0

    start_time = time.time()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break

        grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
        if grab_result and grab_result.GrabSucceeded():
            img = grab_result.Array.copy()
            grab_result.Release()
            # Convert Bayer to RGB
            frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        else:
            continue
        H, W = frame.shape[:2]

        t0 = time.time()
        res = model(frame, verbose=False)[0]
        fps = 1.0 / max(1e-6, time.time() - t0)
        fps_list.append(fps)

        vis = frame.copy()
        if DRAW_MASK_AND_BBOX:
            vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)

        masks = resize_masks_to_frame(res, H, W)
        groove_masks = [m for (cls_id, m) in masks if cls_id == CLASS_GROOVE]
        groove_mask = pick_largest_mask(groove_masks)
        groove_bbox = bbox_from_mask(groove_mask)

        ebox = best_electrode_bbox(res)  # (x1,y1,x2,y2,conf) or None
        electrode_center = None
        mm_per_px = SCALE_MM_PER_PX_MANUAL
        width_px_electrode = None
        if ebox is not None:
            x1,y1,x2,y2,_ = ebox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            electrode_center = (cx, cy)
            width_px_electrode = max(1, x2 - x1)
            if mm_per_px is None:
                mm_per_px = ELECTRODE_DIAMETER_MM / float(width_px_electrode)

        # clearance - extract wrod redius
        clearance_left_mm = None
        clearance_right_mm = None
        center_to_left_mm = None
        center_to_right_mm = None
        groove_left_x = groove_right_x = None

        if (groove_mask is not None) and (electrode_center is not None):
            cx, cy = electrode_center


            xL, xR, y_used = groove_edges_at_y(cy, groove_mask, groove_bbox, USE_GROOVE_BBOX_FOR_EDGES, search=6)
            if (xL is not None) and (xR is not None):
                groove_left_x = xL
                groove_right_x = xR


                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(vis, (xL, cy), 4, (255, 0, 0), -1)
                cv2.circle(vis, (xR, cy), 4, (255, 0, 0), -1)

                if mm_per_px is not None:
                    center_to_left_mm = abs(cx - xL) * mm_per_px
                    center_to_right_mm = abs(xR - cx) * mm_per_px

                    r_px = (ELECTRODE_DIAMETER_MM / mm_per_px) / 2.0
                    clearance_left_mm = max(0.0, (cx - r_px - xL) * mm_per_px)
                    clearance_right_mm = max(0.0, (xR - (cx + r_px)) * mm_per_px)

                draw_hline_with_text(
                    vis, cy, min(cx, xL), max(cx, xL),
                    f"{clearance_left_mm:.2f} mm" if (DRAW_DISTANCE_TEXT and clearance_left_mm is not None) else None,
                    color=(0, 255, 0), thickness=2
                )
                draw_hline_with_text(
                    vis, cy, min(cx, xR), max(cx, xR),
                    f"{clearance_right_mm:.2f} mm" if (DRAW_DISTANCE_TEXT and clearance_right_mm is not None) else None,
                    color=(0, 255, 0), thickness=2
                )

        cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.imshow("YOLO Real-time Prediction", cv2.resize(vis, None, fx=1, fy=1))  # fx=0.6, fy=0.6

        cur_ms = (time.time() - start_time) * 1000.0
        if cur_ms >= next_record_ms:
            record.update({
                "timestamp_ms": float(cur_ms),
                "mm_per_px": float(mm_per_px) if mm_per_px is not None else None,
                "electrode_center_px": list(electrode_center) if electrode_center else None,
                "electrode_bbox_px": list(ebox[:4]) if ebox else None,
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

    camera.StopGrabbing()
    camera.Close()

    if fps_list:
        print(f"Average FPS: {sum(fps_list)/len(fps_list):.2f}")


if __name__ == "__main__":
    predict_video()
