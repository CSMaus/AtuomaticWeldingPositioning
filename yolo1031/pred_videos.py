import os, time, cv2, numpy as np
from ultralytics import YOLO
import torch

CLASS_GROOVE = 0
CLASS_WROD   = 1
CONF = 0.56
ELECTRODE_DIAMETER_MM = 4.3

INFER_W, INFER_H = 640, 360


def bbox_from_mask(mask):
    if mask is None or mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def predict_video():
    script = os.path.dirname(os.path.abspath(__file__))
    # videos_dir = r"D:\ML_DL_AI_stuff\!!DoosanWelding2025\code\AtuomaticWeldingPositioning\DS_collection\original_video"
    videos_dir = r"C:\Users\oem\Documents\GitHub\AtuomaticWeldingPositioning\DS_collection\original_video"

    vids = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not vids:
        print("No videos found")
        return

    for i, v in enumerate(vids):
        print(f"{i}: {v}")
    idx = int(input("Enter video index: "))
    path = os.path.join(videos_dir, vids[idx])

    model_path = os.path.join(script, "runs", "segment", "welding_seg_1203-", "weights", "best.pt")
    model = YOLO(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.fuse()
    if device == "cuda":
        model.model.half()

    dummy = np.zeros((INFER_H, INFER_W, 3), dtype=np.uint8)
    _ = model(dummy, conf=CONF, verbose=False)

    cap = cv2.VideoCapture(path)
    paused = False

    # previous centers in SMALL coords
    prev_cx_groove = {"left": None, "right": None}
    prev_cx_rod    = {"left": None, "right": None}

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

        H_full, W_full = frame.shape[:2]

        t0 = time.time()

        small = cv2.resize(frame, (INFER_W, INFER_H))
        Hs, Ws = small.shape[:2]
        mid = Ws // 2

        res = model(small, conf=CONF, verbose=False)[0]

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.detach().cpu().numpy()
            cls   = res.boxes.cls.detach().cpu().numpy().astype(int)
            confs = res.boxes.conf.detach().cpu().numpy()
        else:
            boxes = np.zeros((0, 4), dtype=float)
            cls   = np.zeros((0,), dtype=int)
            confs = np.zeros((0,), dtype=float)

        fps = 1.0 / max(1e-6, time.time() - t0)

        groove_idx   = {"left": None, "right": None}
        groove_score = {"left": -1e9, "right": -1e9}

        if res.masks is not None and len(res.masks.data) > 0:
            for i, (cid, c) in enumerate(zip(cls, confs)):
                if cid != CLASS_GROOVE:
                    continue
                x1, y1, x2, y2 = boxes[i].astype(int)
                cx = (x1 + x2) // 2
                side = "left" if cx < mid else "right"

                if prev_cx_groove[side] is None:
                    score = c
                else:
                    d = abs(cx - prev_cx_groove[side])
                    score = 0.7 * c - 0.3 * d

                if score > groove_score[side]:
                    groove_score[side] = score
                    groove_idx[side] = i
                    prev_cx_groove[side] = cx

        groove_mask = {"left": None, "right": None}
        if res.masks is not None and len(res.masks.data) > 0:
            for side in ["left", "right"]:
                idx = groove_idx[side]
                if idx is None:
                    continue
                m_raw = res.masks.data[idx].detach().cpu().numpy()
                m_resized = cv2.resize(m_raw, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
                m8 = (m_resized * 255).astype(np.uint8)
                groove_mask[side] = m8

        electrode      = {"left": None, "right": None}
        electrode_conf = {"left": -1, "right": -1}
        rod_idx        = {"left": None, "right": None}

        for i, (box, cid, c) in enumerate(zip(boxes, cls, confs)):
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
                rod_idx[side] = i
                prev_cx_rod[side] = cx

        rod_mask = {"left": None, "right": None}
        if res.masks is not None and len(res.masks.data) > 0:
            for side in ["left", "right"]:
                idx = rod_idx[side]
                if idx is None:
                    continue
                m_raw = res.masks.data[idx].detach().cpu().numpy()
                m_resized = cv2.resize(m_raw, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
                m8 = (m_resized * 255).astype(np.uint8)
                rod_mask[side] = m8

        vis_small = small.copy()

        groove_bbox_s = {"left": None, "right": None}
        rod_bbox_s    = {"left": None, "right": None}
        dist_info     = {"left": None, "right": None}

        for side in ["left", "right"]:
            m = groove_mask[side]
            if m is not None:
                gcol = np.zeros_like(vis_small)
                gcol[:, :, 0] = (m * (150 / 255)).astype(np.uint8)  # B
                gcol[:, :, 1] = (m * (220 / 255)).astype(np.uint8)  # G
                gcol[:, :, 2] = (m * (0   / 255)).astype(np.uint8)  # R
                vis_small = cv2.addWeighted(vis_small, 1.0, gcol, 0.15, 0)

        for side in ["left", "right"]:
            m = rod_mask[side]
            if m is not None:
                rcol = np.zeros_like(vis_small)
                rcol[:, :, 2] = m  # R
                vis_small = cv2.addWeighted(vis_small, 1.0, rcol, 0.15, 0)

        for side in ["left", "right"]:
            m = groove_mask[side]
            if m is None:
                continue
            bb = bbox_from_mask(m)
            if bb:
                x1, y1, x2, y2 = bb
                groove_bbox_s[side] = (x1, y1, x2, y2)
                cv2.rectangle(vis_small, (x1, y1), (x2, y2), (150, 220, 0), 2)

        for side in ["left", "right"]:
            box = electrode[side]
            if box:
                x1, y1, x2, y2 = box
                rod_bbox_s[side] = (x1, y1, x2, y2)
                cv2.rectangle(vis_small, (x1, y1), (x2, y2), (0, 0, 200), 2)

        for side in ["left", "right"]:
            gbb = groove_bbox_s[side]
            rbb = rod_bbox_s[side]
            if gbb is None or rbb is None:
                continue

            gx1, gy1, gx2, gy2 = gbb
            x1,  y1,  x2,  y2  = rbb
            cx_s = (x1 + x2) // 2
            cy_s = (y1 + y2) // 2

            if side == "right":
                xL_s = gx2
            else:
                xL_s = gx1

            px = max(1, x2 - x1)
            mm_per_px = ELECTRODE_DIAMETER_MM / float(px)
            dist_mm = abs(cx_s - xL_s) * mm_per_px

            dist_info[side] = {
                "cx_s": cx_s,
                "cy_s": cy_s,
                "xL_s": xL_s,
                "dist_mm": dist_mm,
                "rod_bbox": rbb,
                "groove_bbox": gbb,
            }

        vis = cv2.resize(vis_small, (W_full, H_full))
        scale_x_full = W_full / float(Ws)
        scale_y_full = H_full / float(Hs)

        for side in ["left", "right"]:
            info = dist_info[side]
            if info is None:
                continue

            cx_s = info["cx_s"]
            cy_s = info["cy_s"]
            xL_s = info["xL_s"]
            dist_mm = info["dist_mm"]
            rbb = info["rod_bbox"]
            gbb = info["groove_bbox"]

            CX = int(round(cx_s * scale_x_full))
            CY = int(round(cy_s * scale_y_full))
            XL = int(round(xL_s * scale_x_full))

            GY = CY

            cv2.circle(vis, (CX, CY), 5, (0, 0, 255), -1)
            cv2.circle(vis, (XL, GY), 5, (255, 0, 0), -1)
            cv2.line(vis, (XL, GY), (CX, CY), (0, 255, 0), 2)

            tx = (CX + XL) // 2
            ty = max(CY - 10, 0)
            cv2.putText(vis, f"{dist_mm:.2f} mm", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            gx1_s, gy1_s, gx2_s, gy2_s = gbb
            rx1_s, ry1_s, rx2_s, ry2_s = rbb

            GX1 = int(round(gx1_s * scale_x_full))
            GY2 = int(round(gy2_s * scale_y_full))
            RX1 = int(round(rx1_s * scale_x_full))
            RY1 = int(round(ry1_s * scale_y_full))

            cv2.putText(vis, "Groove", (GX1, GY2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 220, 0), 2)
            cv2.putText(vis, "W-Rod", (RX1, RY1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO dual-groove", vis)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_video()
