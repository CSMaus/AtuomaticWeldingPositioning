import cv2
import numpy as np
import heapq

def compute_forward_energy(gray_img):
    h, w = gray_img.shape
    energy = np.zeros((h, w), dtype=np.float64)

    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    base_energy = np.abs(sobel_x) + np.abs(sobel_y)

    m = np.copy(base_energy)

    for i in range(1, h):
        for j in range(w):
            left = base_energy[i - 1, j - 1] if j - 1 >= 0 else float('inf')
            up = base_energy[i - 1, j]
            right = base_energy[i - 1, j + 1] if j + 1 < w else float('inf')

            cost_left = left + abs(int(gray_img[i, j]) - int(gray_img[i, j - 1])) if j - 1 >= 0 else float('inf')
            cost_up = up
            cost_right = right + abs(int(gray_img[i, j]) - int(gray_img[i, j + 1])) if j + 1 < w else float('inf')

            m[i, j] += min(cost_left, cost_up, cost_right)

    return m

def find_seam_dijkstra(energy):
    h, w = energy.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w), -1, dtype=np.int32)

    pq = [(energy[0, x], 0, x) for x in range(w)]
    for _, i, j in pq:
        dist[i, j] = energy[i, j]

    heapq.heapify(pq)

    while pq:
        cost, i, j = heapq.heappop(pq)
        if i == h - 1:
            break

        for dj in [-1, 0, 1]:
            nj = j + dj
            if 0 <= nj < w:
                ni = i + 1
                new_cost = cost + energy[ni, nj]
                if new_cost < dist[ni, nj]:
                    dist[ni, nj] = new_cost
                    prev[ni, nj] = j
                    heapq.heappush(pq, (new_cost, ni, nj))

    end_col = np.argmin(dist[h - 1])
    seam = [(end_col, h - 1)]
    for i in range(h - 1, 0, -1):
        end_col = prev[i, end_col]
        seam.append((end_col, i - 1))

    seam.reverse()
    return seam



from collections import deque
import numpy as np

# Initialize buffer to hold groove center points across last 5 frames
groove_lines_buffer = deque(maxlen=5)  # stores lists of points for each frame

def smooth_groove_line(new_line, alpha=0.8):
    """
    Smooths the groove line over the last N frames.
    `new_line` is expected to be a list of (x, y) tuples.
    """
    groove_lines_buffer.append(np.array(new_line, dtype=np.float32))

    if len(groove_lines_buffer) == 1:
        return new_line  # no smoothing possible yet

    # Ensure same y-values across all lines
    ref_y = groove_lines_buffer[0][:, 1]
    smoothed_line = np.zeros_like(groove_lines_buffer[0])

    for i in range(len(ref_y)):
        x_vals = np.array([line[i][0] for line in groove_lines_buffer])
        y_val = ref_y[i]
        x_smooth = np.average(x_vals, weights=[alpha**i for i in range(len(x_vals))[::-1]])
        smoothed_line[i] = (x_smooth, y_val)

    return smoothed_line.astype(int).tolist()



