import os, time, cv2, numpy as np
from ultralytics import YOLO

# ========= USER TOGGLES =========
DRAW_MASK_AND_BBOX   = True   # show YOLO mask+bbox overlay
DRAW_DISTANCE_TEXT   = True   # show distance values on the lines
SCALE_MM_PER_PX_MANUAL = None  # set known scale, else None
ELECTRODE_DIAMETER_MM = 4      # rod diameter (mm)

# ========= HELPERS =========
def largest_mask_bin(masks: np.ndarray) -> np.ndarray:
    """Pick largest segmentation mask (HxW bool/0-255)."""
    if masks is None or len(masks) == 0: return None
    areas = [(m > 0).sum() for m in masks]
    idx = int(np.argmax(areas))
    return (masks[idx] > 0).astype(np.uint8)

def edge_points_at_y(bin_mask: np.ndarray, y: int):
    """Return leftmost and rightmost (x,y) points of mask at row y."""
    H, W = bin_mask.shape
    if y < 0 or y >= H: return None, None
    xs = np.where(bin_mask[y] > 0)[0]
    if len(xs) == 0: return None, None
    return (int(xs.min()), int(y)), (int(xs.max()), int(y))

def draw_line_with_text(img, p1, p2, txt=None, color=(0,255,0), thickness=2):
    cv2.line(img, p1, p2, color, thickness)
    if txt is not None:
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2 - 6)
        cv2.putText(img, txt, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def mm_per_px_from_bbox(bbox, electrode_diam_mm):
    """Estimate mm/px from electrode bbox height or width."""
    if bbox is None: return None
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    ele_px = max(w,h)
    if ele_px <= 0: return None
    return electrode_diam_mm / ele_px

# ========= MAIN =========
def predict_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    videos_dir = os.path.join(project_dir, "data", "basler_recordings")

    vids = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4','.avi','.mov'))]
    if not vids:
        print("No videos found!"); return
    for i,v in enumerate(vids): print(f"{i}: {v}")
    idx = int(input("Enter video index: "))
    video_path = os.path.join(videos_dir, vids[idx])

    model_path = os.path.join(script_dir, "runs", "segment", "weld_seg_0910_1-", "weights", "best.pt")
    if not os.path.exists(model_path):
        print("Trained model not found!"); return
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    paused, fps_list = False, []

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        elif k == ord(' '): paused = not paused
        if paused: continue

        ok, frame = cap.read()
        if not ok: break
        t0 = time.time()
        res = model(frame, verbose=False)[0]
        fps = 1.0 / max(1e-6, (time.time() - t0))
        fps_list.append(fps)

        H,W = frame.shape[:2]
        groove_mask = None
        electrode_center = None
        electrode_bbox = None

        # --- parse results ---
        if res.masks is not None:
            raw_masks = res.masks.data.cpu().numpy()
            mask_resized = [cv2.resize((m*255).astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST) for m in raw_masks]
            classes = res.boxes.cls.cpu().numpy().astype(int)
            if len(classes) == len(mask_resized):
                for cls_id, m in zip(classes, mask_resized):
                    if cls_id == 0:  # groove
                        groove_mask = (m > 0).astype(np.uint8)

        if res.boxes is not None:
            for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
                if cls_id == 1:  # electrode
                    x1,y1,x2,y2 = box.astype(int)
                    electrode_bbox = (x1,y1,x2,y2)
                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)
                    electrode_center = (cx, cy)

        # --- optional draw mask+bbox ---
        if DRAW_MASK_AND_BBOX:
            frame = cv2.addWeighted(frame, 0.6, res.plot(), 0.4, 0)

        # --- compute mm/px ---
        if SCALE_MM_PER_PX_MANUAL is None:
            mm_per_px = mm_per_px_from_bbox(electrode_bbox, ELECTRODE_DIAMETER_MM)
        else:
            mm_per_px = SCALE_MM_PER_PX_MANUAL

        # --- distances: horizontal from electrode center to groove edges ---
        if groove_mask is not None and electrode_center is not None:
            cx, cy = electrode_center
            left_pt, right_pt = edge_points_at_y(groove_mask, cy)
            if left_pt and right_pt:
                cv2.circle(frame, left_pt, 4, (255,0,0), -1)
                cv2.circle(frame, right_pt,4, (255,0,0), -1)
                cv2.circle(frame, electrode_center,5,(0,0,255),-1)

                # distances
                def dist_mm(p):
                    if mm_per_px is None: return None
                    d_px = abs(p[0]-cx)
                    return max(0.0, d_px*mm_per_px - ELECTRODE_DIAMETER_MM/2.0)

                dL = dist_mm(left_pt)
                dR = dist_mm(right_pt)

                draw_line_with_text(frame, electrode_center, left_pt,
                    f"{dL:.2f} mm" if (DRAW_DISTANCE_TEXT and dL is not None) else None,
                    color=(0,255,0), thickness=2)
                draw_line_with_text(frame, electrode_center, right_pt,
                    f"{dR:.2f} mm" if (DRAW_DISTANCE_TEXT and dR is not None) else None,
                    color=(0,255,0), thickness=2)

        # FPS + show
        cv2.putText(frame, f'FPS: {fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        vis = cv2.resize(frame, None, fx=0.6, fy=0.6)
        cv2.imshow("YOLO Real-time Prediction", vis)

    cap.release()
    cv2.destroyAllWindows()
    if fps_list:
        print(f"Average FPS: {sum(fps_list)/len(fps_list):.2f}")

if __name__ == "__main__":
    predict_video()
