#!/usr/bin/env python3
"""
Ring/Ellipse detector + grid + error offsets (interactive)

Features
--------
1) **Interactive setup** (prompts):
   - Image filename (must be in the same folder as this script; extension optional)
   - Physical size of the square image (mm)
   - Ring diameter (mm)
   - Cell size (mm)  -> First grid offset is set to **cell_size / 2**
   - Ring count      -> Either "rows x cols" (e.g., 33x33) or a total (e.g., 1089)

2) **Detection pipeline**:
   - Enhance image (DoG + CLAHE + unsharp)
   - Canny edges + small morph close
   - Fit ellipses on ring-like contours (inner/outer rims)
   - Merge inner+outer per ring via DBSCAN -> one **center** + one **outer ellipse** per ring

3) **Grid & annotations**:
   - Use the **center ring** (closest to image center) as the grid origin
   - Draw grid: first lines at **cell_size/2** mm from center, then every **cell_size** mm
   - Highlight rings (outer ellipse) in **cyan**
   - Crosshairs: **yellow** for center ring, **red** for others

4) **Error offsets**:
   - Order detected centers **row-major** (top-left → bottom-right)
   - Build **ideal** cell midpoints in the same row-major order
   - Error = detected - ideal (x,y) in **px** and **mm**; plus the Euclidean **distance**
   - The center ring has zero error by construction (grid anchored to it)

Outputs
-------
- Saved under:  ~/Downloads/rings_output_<timestamp>  (Windows/macOS/Linux)
- grid_rings_crosshairs.png      : overlay (grid, rings, crosshairs)
- preview_small.jpg              : downsized overlay
- centers.csv                    : detected centers (row-major)
- ellipses.csv                   : outer ellipse per ring
- errors.csv                     : per-ring errors (px & mm, with distance norms)
- run_config.json                : the interactive inputs you provided

Notes
-----
- Assumes the image is **square** (width ≈ height). px/mm is computed from **width / physical_size_mm**.
- If your ring grid is not exactly N×M after detection, the script continues but prints a warning.
"""

import json
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans


# ------------------------- Utility: Downloads folder -------------------------

def default_downloads_dir() -> Path:
    """Return a folder in the user's Downloads with a timestamp."""
    home = Path.home()
    downloads = home / "Downloads"
    if not downloads.exists():
        downloads = home
    ts = time.strftime("%Y%m%d_%H%M%S")
    return downloads / f"rings_output_{ts}"


# ------------------------- Interactive prompts -------------------------

def prompt_str(prompt: str, default: str | None = None) -> str:
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        if not s and default is not None:
            return default
        if s:
            return s

def prompt_float(prompt: str, default: float) -> float:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("  Please enter a number.")

def parse_grid_count(s: str) -> Tuple[int, int]:
    """
    Accepts "rows x cols" (e.g., 33x33, 33×33, 33*33, 33,33) OR a single total (e.g., 1089).
    If total is a perfect square, use sqrt for rows/cols; otherwise, prompt again.
    """
    sep_chars = ("x", "X", "×", "*", ",")
    for ch in sep_chars:
        if ch in s:
            a, b = s.split(ch, 1)
            r, c = int(a.strip()), int(b.strip())
            if r > 0 and c > 0:
                return r, c
            raise ValueError("Rows and cols must be positive integers.")
    # no separator -> try total
    total = int(s.strip())
    root = int(round(total ** 0.5))
    if root * root == total:
        return root, root
    raise ValueError("Provide rows x cols (e.g., 33x33) or a perfect-square total (e.g., 1089).")

def prompt_grid_rc(default_text: str = "33x33") -> Tuple[int, int]:
    while True:
        s = input(f"How many rings? Enter rows x cols or total [{default_text}]: ").strip()
        if not s:
            s = default_text
        try:
            r, c = parse_grid_count(s)
            return r, c
        except Exception as e:
            print(f"  {e}")


# ------------------------- Image enhancements + detection -------------------------

def enhance_for_edges(gray: np.ndarray) -> np.ndarray:
    """Difference-of-Gaussians + CLAHE + light unsharp to boost ring rims."""
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.6)
    g2 = cv2.GaussianBlur(gray, (0, 0), 12.0)
    dog = cv2.normalize(cv2.subtract(g1, g2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dog = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(dog)
    sharp = cv2.addWeighted(dog, 1.4, cv2.GaussianBlur(dog, (0, 0), 1.2), -0.4, 0)
    return sharp

def find_ellipses(gray: np.ndarray, px_per_mm: float, ring_diam_mm: float) -> list:
    """
    Fit ellipses to ring-like contours.
    Returns list of tuples: [ ((cx,cy),(MA,ma),angle_deg), ... ]
    """
    enh = enhance_for_edges(gray)
    edges = cv2.Canny(enh, 60, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    expected_diam_px = px_per_mm * ring_diam_mm       # e.g., 3 mm * ~px/mm
    # Slack for inner/outer rims and distortion/tilt:
    dmin, dmax = 0.6 * expected_diam_px, 1.7 * expected_diam_px

    ellipses = []
    for cnt in contours:
        if len(cnt) < 20:
            continue
        area = abs(cv2.contourArea(cnt))
        if area < 40 or area > 4000:
            continue
        try:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        if dmin <= ma <= dmax and dmin <= MA <= (dmax * 1.3):
            ellipses.append(((float(cx), float(cy)), (float(MA), float(ma)), float(ang)))
    return ellipses

def merge_rims(ellipses: list, eps_px: float, expect: int):
    """
    Merge inner+outer rim fits via DBSCAN → one center + one 'outer' ellipse per ring.
    Returns:
      centers: (N,2) float32, one per ring
      best_ellipses: list of outer ellipses (or the larger one) per ring
    """
    if not ellipses:
        return np.empty((0, 2), np.float32), []

    centers_raw = np.array([[e[0][0], e[0][1]] for e in ellipses], dtype=np.float32)
    def cluster(eps):
        db = DBSCAN(eps=eps, min_samples=1).fit(centers_raw)
        labels = db.labels_
        return labels, labels.max() + 1

    labels, ncl = cluster(eps_px)
    if ncl != expect:
        for eps_try in (max(1.0, eps_px * 0.8), eps_px * 1.1, eps_px * 1.25, eps_px * 1.5):
            labels, ncl = cluster(eps_try)
            if ncl == expect:
                break

    centers = []
    best_ellipses = [None] * ncl
    for k in range(ncl):
        idx = np.where(labels == k)[0]
        c = centers_raw[idx].mean(axis=0)
        centers.append(c)
        # choose the larger ellipse in the cluster as the ring outline
        best, best_sum = None, -1e9
        for j in idx:
            (cx, cy), (MA, ma), ang = ellipses[j]
            s = MA + ma
            if s > best_sum:
                best_sum = s
                best = ((cx, cy), (MA, ma), ang)
        best_ellipses[k] = best

    return np.array(centers, dtype=np.float32), best_ellipses


# ------------------------- Ordering, ideal grid, and errors -------------------------

def order_centers_row_major(centers: np.ndarray, n_rows: int, n_cols: int, random_state=0):
    """
    Order centers in row-major (top-left -> bottom-right).
    Uses KMeans across y to get 'n_rows', then sorts by x within each row.
    """
    y = centers[:, 1:2]
    km = KMeans(n_clusters=n_rows, n_init=5, random_state=random_state)
    labels = km.fit_predict(y)
    row_means = np.array([y[labels == r].mean() for r in range(n_rows)])
    row_order = np.argsort(row_means)
    label_to_row = {int(lbl): int(i) for i, lbl in enumerate(row_order)}
    rows = np.array([label_to_row[int(lbl)] for lbl in labels], dtype=int)

    # sanity check and fallback if row counts drift too much
    counts = np.array([np.sum(rows == r) for r in range(n_rows)])
    if np.any(counts < n_cols - 2) or np.any(counts > n_cols + 2):
        idx_sorted_by_y = np.argsort(centers[:, 1])
        chunks = np.array_split(idx_sorted_by_y, n_rows)
        rows = np.empty(len(centers), dtype=int)
        for r, chunk in enumerate(chunks):
            rows[chunk] = r

    cols = np.empty(len(centers), dtype=int)
    order = []
    for r in range(n_rows):
        idx = np.where(rows == r)[0]
        idx_sorted = idx[np.argsort(centers[idx, 0])]
        for c, i in enumerate(idx_sorted):
            cols[i] = c
            order.append(i)
    return np.array(order, dtype=int), rows, cols

def build_ideal_grid_centers(cx0, cy0, px_per_mm, rows, cols, row_c, col_c, cell_mm: float):
    """
    Ideal midpoint for (row, col) anchored so that (row_c, col_c) is exactly at (cx0, cy0).
    """
    cell_px = cell_mm * px_per_mm
    dx = (cols - col_c) * cell_px
    dy = (rows - row_c) * cell_px
    ideal_x = cx0 + dx
    ideal_y = cy0 + dy
    return ideal_x, ideal_y

def compute_errors(centers, ideal_x, ideal_y, px_per_mm):
    err_x_px = centers[:, 0] - ideal_x
    err_y_px = centers[:, 1] - ideal_y
    err_norm_px = np.sqrt(err_x_px**2 + err_y_px**2)
    err_x_mm = err_x_px / px_per_mm
    err_y_mm = err_y_px / px_per_mm
    err_norm_mm = err_norm_px / px_per_mm
    return err_x_px, err_y_px, err_norm_px, err_x_mm, err_y_mm, err_norm_mm


# ------------------------- Drawing helpers -------------------------

def draw_mm_grid(img_bgr, cx, cy, px_per_mm, cell_mm: float, color=(0, 255, 0), thickness=1):
    """
    Draw grid lines:
      - first lines at **cell_mm / 2** from center (both axes),
      - then every **cell_mm**.
    """
    h, w = img_bgr.shape[:2]
    first_off_px = (cell_mm * px_per_mm) / 2.0
    cell_px = cell_mm * px_per_mm

    # vertical right
    x = cx + first_off_px
    while x < w:
        cv2.line(img_bgr, (int(round(x)), 0), (int(round(x)), h - 1), color, thickness)
        x += cell_px
    # vertical left
    x = cx - first_off_px
    while x >= 0:
        cv2.line(img_bgr, (int(round(x)), 0), (int(round(x)), h - 1), color, thickness)
        x -= cell_px
    # horizontal down
    y = cy + first_off_px
    while y < h:
        cv2.line(img_bgr, (0, int(round(y))), (w - 1, int(round(y))), color, thickness)
        y += cell_px
    # horizontal up
    y = cy - first_off_px
    while y >= 0:
        cv2.line(img_bgr, (0, int(round(y))), (w - 1, int(round(y))), color, thickness)
        y -= cell_px

def draw_cross(img_bgr, x, y, color, size=12, thickness=1):
    cv2.drawMarker(img_bgr, (int(round(x)), int(round(y))), color,
                   markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)


# ------------------------- Main program -------------------------

def main():
    print("\n=== Ring/Ellipse Detector (interactive) ===")
    print("Place your image in the SAME folder as this script.")
    script_dir = Path(__file__).parent

    # --- Interactive inputs ---
    # Image filename (same folder). If no extension, default to .jpg
    fname = prompt_str("Image filename (same folder, e.g., 'circles_modified.jpg' or 'img_054')", "circles_modified.jpg")
    if "." not in Path(fname).name:
        fname = f"{fname}.jpg"
    image_path = script_dir / fname
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found in script folder: {image_path}")

    # Physical size of the (square) image in mm
    physical_size_mm = prompt_float("Physical image size (mm, width = height)", 148.5)

    # Ring diameter (mm)
    ring_diam_mm = prompt_float("Ring (circle) diameter in mm", 3.0)

    # Cell size (mm) -> grid first offset = cell_size / 2
    cell_mm = prompt_float("Cell size (mm) (grid spacing)", 4.5)

    # Ring count (rows x cols or total)
    rows, cols = prompt_grid_rc("33x33")
    expected_count = rows * cols

    # --- Load image ---
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"Could not read image: {image_path}"
    h, w = gray.shape[:2]
    px_per_mm = w / float(physical_size_mm)

    # --- Detect + merge ---
    ellipses = find_ellipses(gray, px_per_mm, ring_diam_mm)
    centers, outer_ellipses = merge_rims(ellipses, eps_px=4.0, expect=expected_count)
    if len(centers) == 0:
        print("No rings detected. Check image and parameters.")
        return
    if len(centers) != expected_count:
        print(f"[WARN] Detected {len(centers)} rings, expected {expected_count}.")

    # Center ring (closest to image center)
    cx_img, cy_img = w / 2.0, h / 2.0
    cidx = int(np.argmin((centers[:, 0] - cx_img) ** 2 + (centers[:, 1] - cy_img) ** 2))
    cx0, cy0 = centers[cidx]

    # --- Row-major ordering, ideal grid, and errors ---
    order, rm_rows, rm_cols = order_centers_row_major(centers, n_rows=rows, n_cols=cols, random_state=0)
    row_c, col_c = int(rm_rows[cidx]), int(rm_cols[cidx])

    ideal_x, ideal_y = build_ideal_grid_centers(cx0, cy0, px_per_mm, rm_rows, rm_cols, row_c, col_c, cell_mm)
    err_x_px, err_y_px, err_norm_px, err_x_mm, err_y_mm, err_norm_mm = compute_errors(
        centers, ideal_x, ideal_y, px_per_mm
    )

    # --- Output folder ---
    outdir = default_downloads_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Draw: grid (based on cell_mm/2 offset), rings, crosshairs ---
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw_mm_grid(overlay, cx0, cy0, px_per_mm, cell_mm=cell_mm, color=(0, 255, 0), thickness=1)

    for e in outer_ellipses:
        if e is None:
            continue
        cv2.ellipse(overlay, e, (255, 255, 0), 1)  # cyan

    for i, (x, y) in enumerate(centers):
        color = (0, 255, 255) if i == cidx else (0, 0, 255)  # yellow/red
        draw_cross(overlay, x, y, color, size=12, thickness=1)

    # --- Save images ---
    full_path = outdir / "grid_rings_crosshairs.png"
    cv2.imwrite(str(full_path), overlay)
    scale = 1024 / max(h, w)
    preview = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(outdir / "preview_small.jpg"), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # --- Save data (row-major order) ---
    det_ordered = centers[order]
    pd.DataFrame({"row": rm_rows[order], "col": rm_cols[order], "x": det_ordered[:, 0], "y": det_ordered[:, 1]})\
        .to_csv(outdir / "centers.csv", index=False)

    rows_ell = []
    for e in outer_ellipses:
        if e is None:
            rows_ell.append([np.nan] * 5)
        else:
            (ex, ey), (MA, ma), ang = e
            rows_ell.append([ex, ey, MA, ma, ang])
    pd.DataFrame(rows_ell, columns=["cx", "cy", "major", "minor", "angle_deg"]).to_csv(outdir / "ellipses.csv", index=False)

    pd.DataFrame({
        "row": rm_rows[order],
        "col": rm_cols[order],
        "det_x": centers[order, 0],
        "det_y": centers[order, 1],
        "ideal_x": ideal_x[order],
        "ideal_y": ideal_y[order],
        "err_x_px": err_x_px[order],
        "err_y_px": err_y_px[order],
        "err_norm_px": err_norm_px[order],
        "err_x_mm": err_x_mm[order],
        "err_y_mm": err_y_mm[order],
        "err_norm_mm": err_norm_mm[order],
    }).to_csv(outdir / "errors.csv", index=False)

    # --- Save run config (so you have a record of inputs) ---
    config = {
        "image_filename": str(image_path.name),
        "physical_size_mm": physical_size_mm,
        "ring_diam_mm": ring_diam_mm,
        "cell_mm": cell_mm,
        "rows": rows,
        "cols": cols,
        "expected_count": expected_count,
        "px_per_mm": float(px_per_mm),
        "center_ring_index": int(cidx),
        "center_ring_row_col": [int(row_c), int(col_c)],
    }
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("\n=== Done ===")
    print("Image :", image_path)
    print("Saved :", outdir)
    print("  - grid_rings_crosshairs.png")
    print("  - preview_small.jpg")
    print("  - centers.csv")
    print("  - ellipses.csv")
    print("  - errors.csv")
    print("  - run_config.json")
    if len(centers) != expected_count:
        print(f"[WARN] Detected {len(centers)} rings, expected {expected_count}.")

if __name__ == "__main__":
    main()
