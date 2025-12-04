import os, time, cv2, numpy as np
from ultralytics import YOLO

CLASS_GROOVE = 0
CLASS_WROD = 1
CONF = 0.56
ELECTRODE_DIAMETER_MM = 4.3

def bbox_from_mask(mask):
    if mask is None or mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def resize_masks(res, H, W):
    if res.masks is None:
        return []
    raw = res.masks.data.cpu().numpy()
    ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    out = []
    for m, cid, c in zip(raw, ids, confs):
        m8 = (m * 255).astype(np.uint8)
        m8 = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append((cid, m8, c))
    return out

def edge_left(mask, y, search=6):
    H = mask.shape[0]
    for dy in range(search + 1):
        for yy in [y - dy, y + dy] if dy else [y]:
            if 0 <= yy < H:
                xs = np.where(mask[yy] > 0)[0]
                if len(xs):
                    return xs.min(), yy
    return None, None

def edge_right(mask, y, search=6):
    H = mask.shape[0]
    for dy in range(search + 1):
        for yy in [y - dy, y + dy] if dy else [y]:
            if 0 <= yy < H:
                xs = np.where(mask[yy] > 0)[0]
                if len(xs):
                    return xs.max(), yy
    return None, None

def predict_video():
    script = os.path.dirname(os.path.abspath(__file__))
    videos_dir = r"D:\ML_DL_AI_stuff\!!DoosanWelding2025\code\AtuomaticWeldingPositioning\DS_collection\original_video"
    vids = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    for i, v in enumerate(vids):
        print(f"{i}: {v}")
    idx = int(input("Enter video index: "))
    path = os.path.join(videos_dir, vids[idx])

    model_path = os.path.join(script, "runs", "segment", "welding_seg_1203-", "weights", "best.pt")
    model = YOLO(model_path)
    _ = model(np.zeros((512,512,3), dtype=np.uint8), verbose=False)

    cap = cv2.VideoCapture(path)
    paused = False
    prev_cx_groove = {"left": None, "right": None}
    prev_cx_rod = {"left": None, "right": None}

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord(' '):
            paused = not paused
        if paused:
            continue

        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        mid = W // 2

        t0 = time.time()
        res = model(frame, verbose=False, conf=CONF)[0]
        fps = 1.0 / max(1e-6, time.time() - t0)

        masks = resize_masks(res, H, W)
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), int)
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))

        groove_mask = {"left": None, "right": None}
        groove_conf = {"left": -1, "right": -1}

        for cid, m, c in masks:
            if cid != CLASS_GROOVE:
                continue
            bb = bbox_from_mask(m)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            cx = (x1 + x2) // 2
            side = "left" if cx < mid else "right"
            if prev_cx_groove[side] is None:
                score = c
            else:
                d = abs(cx - prev_cx_groove[side])
                score = 0.7 * c - 0.3 * d

            if score > groove_conf[side]:
                groove_conf[side] = score
                groove_mask[side] = m.copy()
                prev_cx_groove[side] = cx

        electrode = {"left": None, "right": None}
        electrode_conf = {"left": -1, "right": -1}

        for box, cid, c in zip(boxes, cls, confs):
            if cid != CLASS_WROD:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cx = (x1 + x2) // 2
            side = "left" if cx < mid else "right"
            if prev_cx_rod[side] is None:
                score = c
            else:
                d = abs(cx - prev_cx_rod[side])
                score = 0.7 * c - 0.3 * d

            if score > electrode_conf[side]:
                electrode_conf[side] = score
                electrode[side] = (x1, y1, x2, y2)
                prev_cx_rod[side] = cx

        vis = frame.copy()
        # vis = cv2.addWeighted(vis, 0.6, res.plot(), 0.4, 0)
        for side in ["left", "right"]:
            m = groove_mask[side]
            if m is not None:
                gcol = np.zeros_like(vis)
                gcol[:, :, 1] = m
                gcol[:, :, 0] = (m * (150 / 255)).astype(np.uint8)  # Blue
                gcol[:, :, 1] = (m * (220 / 255)).astype(np.uint8)  # Green
                gcol[:, :, 2] = (m * (0 / 255)).astype(np.uint8)  # Red
                vis = cv2.addWeighted(vis, 1.0, gcol, 0.15, 0)

        for side in ["left", "right"]:
            box = electrode[side]
            if box is None:
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            rod_mask = None
            for cid, m, c in masks:
                if cid == CLASS_WROD:
                    bb = bbox_from_mask(m)
                    if bb is not None:
                        bx1, by1, bx2, by2 = bb
                        if abs(x1 - bx1) < w and abs(x2 - bx2) < w:
                            rod_mask = m
                            break
            if rod_mask is not None:
                rcol = np.zeros_like(vis)
                rcol[:, :, 2] = rod_mask
                vis = cv2.addWeighted(vis, 1.0, rcol, 0.15, 0)

        for side in ["left", "right"]:
            m = groove_mask[side]
            if m is None:
                continue
            bb = bbox_from_mask(m)
            if bb:
                x1, y1, x2, y2 = bb
                cv2.rectangle(vis, (x1, y1), (x2, y2), (150,220,0), 2)
                cv2.putText(vis, "Groove", (x2-100, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,220, 0), 2)

        for side in ["left", "right"]:
            box = electrode[side]
            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0,0,200), 2)
                cv2.putText(vis, "W-Rod", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

        for side in ["left", "right"]:
            m = groove_mask[side]
            box = electrode[side]
            if m is None or box is None:
                continue
            x1,y1,x2,y2 = box
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            if side == "left":
                m[:, mid:] = 0
                gx, gy = edge_left(m, cy)
            else:
                m[:, :mid] = 0
                gx, gy = edge_right(m, cy)
            if gx is None:
                continue
            px = max(1, x2 - x1)
            mm = ELECTRODE_DIAMETER_MM / float(px)
            dist = abs(cx - gx) * mm
            cv2.circle(vis, (cx,cy), 5, (0,0,255), -1)
            cv2.circle(vis, (gx,cy), 5, (255,0,0), -1)
            cv2.line(vis, (gx,cy), (cx,cy), (0,255,0), 2)
            tx = (cx+gx)//2
            ty = max(cy-10, 0)
            cv2.putText(vis, f"{dist:.2f} mm", (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(vis, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("YOLO dual-groove", vis)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_video()
