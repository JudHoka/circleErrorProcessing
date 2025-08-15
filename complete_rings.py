#!/usr/bin/env python3
"""
Rotation-only ring/ellipse detector and grid overlay + CAL updater.

Pipeline:
  1) Detect rings/ellipses on the raw image (auto-tuned).
  2) Estimate a tiny global rotation that best aligns rows/cols (no scale, no shear).
  3) Rotate the image by that angle (image size stays identical).
  4) Re-detect on the rotated image, build the ideal grid, draw crosshairs + grid,
     and export centers/ellipses/errors.
  5) (Optional) Load an existing .cal and update X_Deviation / Y_Deviation tables
     from the measured Δx/Δy (mm), with small-value tolerance.

NO homography, NO poly2, NO TPS — image geometry is preserved except for a small rotation.

Outputs -> ~/Downloads/rings_output_<timestamp>
    rotated_for_MCP.bmp (convenience export), grid_rings_crosshairs.png, preview_small.jpg,
    centers.csv, ellipses.csv, errors.csv, run_config.json, updated_*.cal (if chosen)
"""

import json
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import xml.etree.ElementTree as ET

# ------------------------- Utilities & I/O -------------------------

def default_downloads_dir() -> Path:
    home = Path.home()
    downloads = home / "Downloads"
    if not downloads.exists():
        downloads = home
    ts = time.strftime("%Y%m%d_%H%M%S")
    return downloads / f"rings_output_{ts}"

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

def prompt_bool(prompt: str, default: bool=False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} [{d}]: ").strip().lower()
        if not s:
            return default
        if s in ("y","yes"): return True
        if s in ("n","no"): return False
        print("  Please answer y or n.")

def parse_grid_count(s: str) -> Tuple[int, int]:
    for ch in ("x", "X", "×", "*", ","):
        if ch in s:
            a, b = s.split(ch, 1)
            r, c = int(a.strip()), int(b.strip())
            if r > 0 and c > 0:
                return r, c
            raise ValueError("Rows and cols must be positive integers.")
    total = int(s.strip())
    root = int(round(total ** 0.5))
    if root * root == total:
        return root, root
    raise ValueError("Provide 'rows x cols' (e.g., 33x33) or a perfect-square total (e.g., 1089).")

def prompt_grid_rc(default_text: str = "33x33") -> Tuple[int, int]:
    while True:
        s = input(f"How many rings? Enter rows x cols or total [{default_text}]: ").strip()
        if not s:
            s = default_text
        try:
            return parse_grid_count(s)
        except Exception as e:
            print(f"  {e}")

def as_nx2(arr) -> np.ndarray:
    """
    Ensure array is float32 of shape (N,2). Accepts lists, 1-D arrays, etc.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 0:
        return np.zeros((0, 2), np.float32)
    if a.ndim == 1:
        if a.size == 0:
            return np.zeros((0, 2), np.float32)
        if a.size % 2 != 0:
            raise ValueError(f"Expected even number of values for centers, got {a.size}.")
        a = a.reshape(-1, 2)
    elif a.ndim == 2 and a.shape[1] != 2:
        a = a.reshape(-1, 2)
    return a.astype(np.float32)

# ------------------------- Enhancement & detection -------------------------

def enhance_for_edges(gray: np.ndarray, dog_sigma2: float) -> np.ndarray:
    """DoG + CLAHE + light unsharp."""
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.6)
    g2 = cv2.GaussianBlur(gray, (0, 0), dog_sigma2)
    dog = cv2.normalize(cv2.subtract(g1, g2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dog = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(dog)
    sharp = cv2.addWeighted(dog, 1.4, cv2.GaussianBlur(dog, (0, 0), 1.2), -0.4, 0)
    return sharp

def find_ellipses(gray: np.ndarray,
                  px_per_mm: float,
                  ring_diam_mm: float,
                  det: Dict[str, Any]) -> list:
    """
    Return: [ ((cx,cy),(MA,ma),angle_deg), ... ]
    Uses dynamic area gating based on expected diameter, plus axis-length gating.
    """
    enh = enhance_for_edges(gray, det["dog_sigma2"])
    edges = cv2.Canny(enh, det["canny_lo"], det["canny_hi"])
    if det["morph"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (det["morph"], det["morph"]))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    expected_diam_px = float(px_per_mm * ring_diam_mm)
    dmin = det["dmin_scale"] * expected_diam_px
    dmax = det["dmax_scale"] * expected_diam_px

    exp_area = np.pi * (0.5 * expected_diam_px) ** 2
    area_min_dyn = 0.20 * exp_area
    area_max_dyn = 5.00 * exp_area

    ellipses = []
    for cnt in contours:
        if len(cnt) < 20:
            continue
        a = abs(cv2.contourArea(cnt))
        if a < area_min_dyn or a > area_max_dyn:
            continue
        try:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        big, small = max(MA, ma), min(MA, ma)
        if dmin <= small <= dmax and dmin <= big <= (dmax * 1.3):
            ellipses.append(((float(cx), float(cy)), (float(MA), float(ma)), float(ang)))
    return ellipses

def merge_rims_with_eps(ellipses: list, eps_px: float):
    """
    DBSCAN-merge inner/outer rims -> one center (cluster mean) + keep the largest ellipse per cluster.
    """
    if not ellipses:
        return np.empty((0, 2), np.float32), []
    centers_raw = np.array([[e[0][0], e[0][1]] for e in ellipses], dtype=np.float32)
    db = DBSCAN(eps=eps_px, min_samples=1).fit(centers_raw)
    labels = db.labels_
    ncl = labels.max() + 1

    centers = np.zeros((ncl, 2), np.float32)
    best_ellipses = [None] * ncl
    for k in range(ncl):
        idx = np.where(labels == k)[0]
        pts = centers_raw[idx]
        centers[k] = pts.mean(axis=0)  # cluster mean
        best, best_sum = None, -1e9
        for j in idx:
            (cx, cy), (MA, ma), ang = ellipses[j]
            s = MA + ma
            if s > best_sum:
                best_sum = s
                best = ((cx, cy), (MA, ma), ang)
        best_ellipses[k] = best
    return centers, best_ellipses

def centers_from_best(outer_ellipses: list, fallback_centers: np.ndarray) -> np.ndarray:
    """Use the drawn ellipse's center; fall back to cluster mean if ellipse missing."""
    fallback_centers = as_nx2(fallback_centers)
    rows = []
    for i, e in enumerate(outer_ellipses):
        if e is not None:
            (cx, cy), _, _ = e
            rows.append([float(cx), float(cy)])
        else:
            if fallback_centers.shape[0] > i:
                rows.append([float(fallback_centers[i, 0]), float(fallback_centers[i, 1])])
    if not rows:
        return np.zeros((0, 2), np.float32)
    return np.vstack(rows).astype(np.float32)

# ------------------------- Ordering, ideal grid, errors -------------------------

def order_centers_row_major(centers: np.ndarray, n_rows: int, n_cols: int, random_state=0):
    centers = as_nx2(centers)
    if centers.shape[0] == 0:
        raise ValueError("order_centers_row_major: empty centers array")

    y = centers[:, 1:2]
    km = KMeans(n_clusters=n_rows, n_init=5, random_state=random_state)
    labels = km.fit_predict(y)
    row_means = np.array([y[labels == r].mean() for r in range(n_rows)])
    row_order = np.argsort(row_means)
    label_to_row = {int(lbl): int(i) for i, lbl in enumerate(row_order)}
    rows = np.array([label_to_row[int(lbl)] for lbl in labels], dtype=int)

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
    cell_px = cell_mm * px_per_mm
    dx = (cols - col_c) * cell_px
    dy = (rows - row_c) * cell_px
    return cx0 + dx, cy0 + dy

def compute_errors(centers, ideal_x, ideal_y, px_per_mm):
    err_x_px = centers[:, 0] - ideal_x
    err_y_px = centers[:, 1] - ideal_y
    err_norm_px = np.sqrt(err_x_px**2 + err_y_px**2)
    err_x_mm = err_x_px / px_per_mm
    err_y_mm = err_y_px / px_per_mm
    err_norm_mm = err_norm_px / px_per_mm
    return err_x_px, err_y_px, err_norm_px, err_x_mm, err_y_mm, err_norm_mm

# ------------------------- Micro-rotation refine (rotation only) -------------------------

def rotate_keep_square(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around center, keep same WxH."""
    if abs(angle_deg) < 1e-9:
        return gray.copy()
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def _row_midpoints(centers: np.ndarray, rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    mids = []
    for r in range(n_rows):
        idxL = np.where((rows == r) & (cols == 0))[0]
        idxR = np.where((rows == r) & (cols == n_cols - 1))[0]
        if len(idxL) and len(idxR):
            L = centers[idxL[0]]; R = centers[idxR[0]]
            mids.append(((L[0] + R[0]) * 0.5, (L[1] + R[1]) * 0.5))
    return np.array(mids, dtype=np.float32)

def _col_midpoints(centers: np.ndarray, rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    mids = []
    for c in range(n_cols):
        idxT = np.where((cols == c) & (rows == 0))[0]
        idxB = np.where((cols == c) & (rows == n_rows - 1))[0]
        if len(idxT) and len(idxB):
            T = centers[idxT[0]]; B = centers[idxB[0]]
            mids.append(((T[0] + B[0]) * 0.5, (T[1] + B[1]) * 0.5))
    return np.array(mids, dtype=np.float32)

def _slopes_cost_for_angle(centers: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                           n_rows: int, n_cols: int, cx_img: float, cy_img: float, angle_deg: float) -> float:
    """Rotate centers by angle and measure how horizontal/vertical the midlines are."""
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    ctr_rot = ((centers - np.array([[cx_img, cy_img]], dtype=np.float32)) @ R.T) + np.array([[cx_img, cy_img]], dtype=np.float32)

    # Rows: fit y ≈ m*x + b → m should be ~0
    row_mids = _row_midpoints(ctr_rot, rows, cols, n_rows, n_cols)
    cost = 0.0
    if len(row_mids) >= 3:
        X, Y = row_mids[:, 0], row_mids[:, 1]
        m_row = np.linalg.lstsq(np.vstack([X, np.ones_like(X)]).T, Y, rcond=None)[0][0]
        cost += abs(m_row)

    # Cols: fit x ≈ a*y + b → a should be ~0
    col_mids = _col_midpoints(ctr_rot, rows, cols, n_rows, n_cols)
    if len(col_mids) >= 3:
        Y, X = col_mids[:, 1], col_mids[:, 0]
        a_col = np.linalg.lstsq(np.vstack([Y, np.ones_like(Y)]).T, X, rcond=None)[0][0]
        cost += abs(a_col)

    return float(cost)

def refine_rotation_small_angle(gray: np.ndarray, centers: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                                n_rows: int, n_cols: int, search_deg: float = 0.8) -> float:
    """
    1D search for tiny angle that minimizes combined slopes of row/col midlines.
    Two-stage: coarse (0.05°), then fine (0.01°) near the best.
    """
    h, w = gray.shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0

    best_ang, best_cost = 0.0, float("inf")
    # coarse
    ang = -search_deg
    while ang <= search_deg + 1e-9:
        c = _slopes_cost_for_angle(centers, rows, cols, n_rows, n_cols, cx_img, cy_img, float(ang))
        if c < best_cost:
            best_cost, best_ang = c, float(ang)
        ang += 0.05

    # fine around best
    lo, hi = best_ang - 0.1, best_ang + 0.1
    ang = lo
    while ang <= hi + 1e-9:
        c = _slopes_cost_for_angle(centers, rows, cols, n_rows, n_cols, cx_img, cy_img, float(ang))
        if c < best_cost:
            best_cost, best_ang = c, float(ang)
        ang += 0.01

    return best_ang

# ------------------------- AUTO-TUNE (with adaptive EPS grid) -------------------------

DETECT_CONFIGS = [
    {"name":"base",   "canny_lo":60, "canny_hi":140, "morph":3, "dog_sigma2":12.0,
     "dmin_scale":0.60, "dmax_scale":1.70, "area_min":1, "area_max":1_000_000},
    {"name":"sens1",  "canny_lo":50, "canny_hi":130, "morph":3, "dog_sigma2":12.0,
     "dmin_scale":0.55, "dmax_scale":1.80, "area_min":1, "area_max":1_000_000},
    {"name":"sens2",  "canny_lo":40, "canny_hi":120, "morph":5, "dog_sigma2":10.0,
     "dmin_scale":0.50, "dmax_scale":2.00, "area_min":1, "area_max":1_000_000},
    {"name":"strict1","canny_lo":70, "canny_hi":160, "morph":3, "dog_sigma2":14.0,
     "dmin_scale":0.65, "dmax_scale":1.60, "area_min":1, "area_max":1_000_000},
    {"name":"strict2","canny_lo":80, "canny_hi":180, "morph":3, "dog_sigma2":16.0,
     "dmin_scale":0.70, "dmax_scale":1.50, "area_min":1, "area_max":1_000_000},
]

def adaptive_eps_grid(px_per_mm: float, ring_diam_mm: float) -> List[float]:
    """Base eps ≈ 25% of ring radius (px), clamped to [1.5, 8.0], then sweep around it."""
    radius_px = 0.5 * ring_diam_mm * px_per_mm
    base = float(np.clip(0.25 * radius_px, 1.5, 8.0))
    grid = [0.6*base, 0.8*base, base, 1.2*base, 1.5*base, 2.0*base, 2.5*base]
    grid = sorted({float(np.clip(g, 1.5, 8.0)) for g in grid})
    return grid

def _best_eps_choice(ellipses: list, expected_count: int, eps_grid: List[float]) -> dict:
    if not ellipses:
        return {"eps": 4.0, "count": 0, "grid": []}
    results = []
    for eps in eps_grid:
        centers, _ = merge_rims_with_eps(ellipses, eps)
        centers = as_nx2(centers)
        results.append((float(eps), int(centers.shape[0])))
    results.sort(key=lambda r: (abs(r[1] - expected_count), r[0]))
    best_eps, best_count = results[0]
    return {"eps": best_eps, "count": best_count, "grid": results}

def _make_trial(det_cfg: dict, best_eps_info: dict, ellipses: list, expected_count: int) -> dict:
    centers, best_ell = merge_rims_with_eps(ellipses, best_eps_info["eps"])
    centers = as_nx2(centers)
    count = int(centers.shape[0])
    diff = abs(count - expected_count)
    return {
        "det": det_cfg,
        "eps": float(best_eps_info["eps"]),
        "count": count,
        "diff": diff,
        "centers": centers,
        "ellipses": best_ell,
        "eps_scan": best_eps_info.get("grid", []),
    }

def autoselect_config_and_eps(gray, px_per_mm, ring_diam_mm, expected_count: int, eps_grid: List[float]) -> dict:
    trials = []

    base = DETECT_CONFIGS[0]
    ell_base = find_ellipses(gray, px_per_mm, ring_diam_mm, base)
    best_for_base = _best_eps_choice(ell_base, expected_count, eps_grid)
    trials.append(_make_trial(base, best_for_base, ell_base, expected_count))

    if trials[-1]["count"] < expected_count and trials[-1]["diff"] > 0:
        direction = "under"
    elif trials[-1]["count"] > expected_count and trials[-1]["diff"] > 0:
        direction = "over"
    else:
        direction = "unknown"

    if direction == "under":
        order = [DETECT_CONFIGS[1], DETECT_CONFIGS[2], DETECT_CONFIGS[3], DETECT_CONFIGS[4]]
    elif direction == "over":
        order = [DETECT_CONFIGS[3], DETECT_CONFIGS[4], DETECT_CONFIGS[1], DETECT_CONFIGS[2]]
    else:
        order = [DETECT_CONFIGS[1], DETECT_CONFIGS[3], DETECT_CONFIGS[2], DETECT_CONFIGS[4]]

    for det in order:
        ell = find_ellipses(gray, px_per_mm, ring_diam_mm, det)
        best = _best_eps_choice(ell, expected_count, eps_grid)
        trials.append(_make_trial(det, best, ell, expected_count))
        if trials[-1]["count"] == expected_count:
            break

    trials.sort(key=lambda t: (t["diff"], 1 if t["count"] >= expected_count else 0))
    return trials[0]

# ------------------------- Drawing -------------------------

def draw_mm_grid(img_bgr, cx, cy, px_per_mm, cell_mm: float, color=(0, 255, 0), thickness=1):
    """First lines at cell_mm/2 from center, then every cell_mm."""
    h, w = img_bgr.shape[:2]
    first_off_px = (cell_mm * px_per_mm) / 2.0
    cell_px = cell_mm * px_per_mm

    x = cx + first_off_px
    while x < w:
        cv2.line(img_bgr, (int(round(x)), 0), (int(round(x)), h - 1), color, thickness)
        x += cell_px
    x = cx - first_off_px
    while x >= 0:
        cv2.line(img_bgr, (int(round(x)), 0), (int(round(x)), h - 1), color, thickness)
        x -= cell_px

    y = cy + first_off_px
    while y < h:
        cv2.line(img_bgr, (0, int(round(y))), (w - 1, int(round(y))), color, thickness)
        y += cell_px
    y = cy - first_off_px
    while y >= 0:
        cv2.line(img_bgr, (0, int(round(y))), (w - 1, int(round(y))), color, thickness)
        y -= cell_px

def draw_cross(img_bgr, x, y, color, size=12, thickness=1):
    cv2.drawMarker(img_bgr, (int(round(x)), int(round(y))), color,
                   markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)

# ------------------------- .cal read / update helpers -------------------------

def _read_cal_tables(cal_path: Path) -> tuple[ET.ElementTree, ET.Element, dict[str, np.ndarray], dict[str, ET.Element]]:
    tree = ET.parse(str(cal_path))
    root = tree.getroot()
    tables: dict[str, np.ndarray] = {}
    nodes: dict[str, ET.Element] = {}
    for tab in root.findall("Table"):
        name = tab.get("Name")
        dim0 = int(tab.get("Dim0", "0"))  # columns
        dim1 = int(tab.get("Dim1", "0"))  # rows
        arr = np.zeros((dim1, dim0), dtype=np.float64)  # arr[row, col]
        for el in tab.findall("Element"):
            c = int(el.get("Column")); r = int(el.get("Row"))
            arr[r, c] = float(el.get("Value"))
        tables[name] = arr
        nodes[name] = tab
    return tree, root, tables, nodes

def _write_cal_table(table_node: ET.Element, arr: np.ndarray) -> None:
    # write back values with 12 decimals to match sample files
    for el in table_node.findall("Element"):
        c = int(el.get("Column")); r = int(el.get("Row"))
        v = float(arr[r, c])
        el.set("Value", f"{v:.12f}")

def update_cal_from_errors(
    cal_in: Path,
    cal_out: Path,
    rows: int,
    cols: int,
    err_table_x_mm: np.ndarray,     # shape [rows, cols]
    err_table_y_mm: np.ndarray,     # shape [rows, cols]
    tol_mm: float = 0.001,
    mode: str = "replace",          # "replace" or "increment"
    compensate_sign: bool = True,   # True: write -error (counteract), False: write +error
) -> None:
    """
    Load an existing .cal (XML), and update X_Deviation and Y_Deviation tables.
    Values are per cell in **mm**.

    tol_mm: values with abs(val) < tol_mm are zeroed.
    mode:   "replace"   -> X/Y_Deviation := (+/-)error (after tolerance)
            "increment" -> X/Y_Deviation += (+/-)error (after tolerance)
    compensate_sign=True applies the negative of measured error (usual calibration).
    """
    tree, root, tables, nodes = _read_cal_tables(cal_in)

    # Build clean arrays (rows x cols)
    ex = np.zeros((rows, cols), dtype=np.float64)
    ey = np.zeros((rows, cols), dtype=np.float64)
    ex[:,:] = np.where(np.abs(err_table_x_mm) < tol_mm, 0.0, err_table_x_mm)
    ey[:,:] = np.where(np.abs(err_table_y_mm) < tol_mm, 0.0, err_table_y_mm)

    if compensate_sign:
        ex = -ex
        ey = -ey

    # Ensure tables exist
    if "X_Deviation" not in tables or "Y_Deviation" not in tables:
        raise ValueError("The .cal file does not contain X_Deviation and Y_Deviation tables.")
    tx, ty = tables["X_Deviation"], tables["Y_Deviation"]

    if tx.shape != ex.shape or ty.shape != ey.shape:
        raise ValueError(f"Shape mismatch: .cal is {tx.shape} but your grid is {ex.shape}. "
                         f"Make sure the row/col count matches.")

    if mode == "replace":
        tx_new = ex
        ty_new = ey
    elif mode == "increment":
        tx_new = tx + ex
        ty_new = ty + ey
    else:
        raise ValueError("mode must be 'replace' or 'increment'")

    # Write back
    _write_cal_table(nodes["X_Deviation"], tx_new)
    _write_cal_table(nodes["Y_Deviation"], ty_new)

    # Update date/time attributes (optional)
    root.set("Date", time.strftime("%Y/%m/%d"))
    root.set("Time", time.strftime("%H:%M:%S"))

    # Save
    cal_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(cal_out), encoding="UTF-8", xml_declaration=True)

# ------------------------- Main (interactive) -------------------------

def main():
    print("\n=== Rotation-Only Ring/Ellipse Detector (interactive) + CAL updater ===")
    print("Place your image in the SAME folder as this script.")
    script_dir = Path(__file__).parent

    # ---- Inputs ----
    fname = prompt_str("Image filename (same folder, e.g., 'circles_modified.jpg' or 'img_054')",
                       "circles_modified.jpg")
    if "." not in Path(fname).name:
        fname = f"{fname}.jpg"
    image_path = script_dir / fname
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found in script folder: {image_path}")

    physical_size_mm = prompt_float("Physical image size (square, mm)", 148.5)
    ring_diam_mm     = prompt_float("Ring (circle) diameter (mm)", 3.0)
    cell_mm          = prompt_float("Cell size (mm) (grid spacing)", 4.5)
    rows, cols       = prompt_grid_rc("33x33")
    expected_count   = rows * cols
    center_mode      = prompt_str("Center mode ('best' uses ellipse center, 'mean' uses cluster mean)", "best").lower()
    if center_mode not in ("best","mean"):
        center_mode = "best"
    debug_diff       = prompt_bool("Draw ellipse-center vs chosen-center debug lines?", False)
    enable_microrot  = prompt_bool("Enable micro-rotation refine (±deg search)?", True)
    microrot_range   = prompt_float("Max rotation search (degrees, small)", 0.8) if enable_microrot else 0.0

    # ---- Load ----
    gray0 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    assert gray0 is not None, f"Could not read image: {image_path}"
    h0, w0 = gray0.shape[:2]
    px_per_mm = w0 / float(physical_size_mm)

    # ---- PASS 1: Auto-tune on RAW image ----
    eps_grid1 = adaptive_eps_grid(px_per_mm, ring_diam_mm)
    trial1 = autoselect_config_and_eps(gray0, px_per_mm, ring_diam_mm, expected_count, eps_grid1)

    centers1_mean, ellipses1 = as_nx2(trial1["centers"]), (trial1["ellipses"] or [])
    print(f"[DEBUG] Pass1 clusters: {trial1['count']}")
    print(f"[DEBUG] Pass1 ellipses list length: {len(ellipses1)}; non-None: {sum(1 for e in ellipses1 if e is not None)}")

    if center_mode == "best" and len(ellipses1) > 0 and any(e is not None for e in ellipses1):
        centers1 = centers_from_best(ellipses1, centers1_mean)
    else:
        if center_mode == "best":
            print("[WARN] Pass1 ellipse list empty/all None; using cluster-mean centers.")
        centers1 = centers1_mean

    if centers1.shape[0] == 0:
        print("[ERROR] No rings detected in Pass 1."); return

    # Order & center ring on raw image
    order1, rows1, cols1 = order_centers_row_major(centers1, n_rows=rows, n_cols=cols, random_state=0)
    cx_img0, cy_img0 = w0/2.0, h0/2.0
    cidx1 = int(np.argmin((centers1[:,0]-cx_img0)**2 + (centers1[:,1]-cy_img0)**2))

    # ---- Rotation-only refine ----
    rotate_deg = 0.0
    gray_rot = gray0
    if enable_microrot:
        rotate_deg = refine_rotation_small_angle(gray0, centers1, rows1, cols1, rows, cols, search_deg=float(microrot_range))
        if abs(rotate_deg) > 0.01:
            gray_rot = rotate_keep_square(gray0, rotate_deg)
            # Re-detect on the rotated image (auto-tune again)
            trial2 = autoselect_config_and_eps(gray_rot, px_per_mm, ring_diam_mm, expected_count,
                                               adaptive_eps_grid(px_per_mm, ring_diam_mm))
            centers2_mean, ellipses2 = as_nx2(trial2["centers"]), (trial2["ellipses"] or [])
            print(f"[DEBUG] Pass2 (rotated) clusters: {trial2['count']}")
            print(f"[DEBUG] Pass2 (rotated) ellipses length: {len(ellipses2)}; non-None: {sum(1 for e in ellipses2 if e is not None)}")
            if center_mode == "best" and len(ellipses2) > 0 and any(e is not None for e in ellipses2):
                centers = centers_from_best(ellipses2, centers2_mean)
                outer_ellipses = ellipses2
            else:
                if center_mode == "best":
                    print("[WARN] Rotated: ellipse list empty/all None; using cluster-mean centers.")
                centers = centers2_mean
                outer_ellipses = ellipses2
        else:
            centers = centers1
            outer_ellipses = ellipses1
    else:
        centers = centers1
        outer_ellipses = ellipses1

    # Rescue if rotated pass failed (rare)
    if centers.shape[0] == 0:
        print("[WARN] Rotated re-detect produced no centers; falling back to pass1 centers.")
        centers = centers1
        outer_ellipses = ellipses1
        gray_rot = gray0
        rotate_deg = 0.0

    # ---- Build ordering on final image & ideal grid ----
    order, rm_rows, rm_cols = order_centers_row_major(centers, n_rows=rows, n_cols=cols, random_state=0)
    h, w = gray_rot.shape[:2]
    cx_img, cy_img = w/2.0, h/2.0
    cidx = int(np.argmin((centers[:,0]-cx_img)**2 + (centers[:,1]-cy_img)**2))

    cx0, cy0 = centers[cidx]
    row_c, col_c = int(rm_rows[cidx]), int(rm_cols[cidx])
    ideal_x, ideal_y = build_ideal_grid_centers(cx0, cy0, px_per_mm, rm_rows, rm_cols, row_c, col_c, cell_mm)

    # ---- Errors ----
    err_x_px, err_y_px, err_norm_px, err_x_mm, err_y_mm, err_norm_mm = compute_errors(
        centers, ideal_x, ideal_y, px_per_mm
    )

    # ---- Output (images, CSVs) ----
    outdir = default_downloads_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    overlay = cv2.cvtColor(gray_rot, cv2.COLOR_GRAY2BGR)
    draw_mm_grid(overlay, cx0, cy0, px_per_mm, cell_mm=cell_mm, color=(0,255,0), thickness=1)

    for i, e in enumerate(outer_ellipses):
        if e is None: continue
        cv2.ellipse(overlay, e, (255,255,0), 1)  # cyan
        if debug_diff:
            (ex, ey), _, _ = e
            cx, cy = centers[i]
            cv2.circle(overlay, (int(round(ex)), int(round(ey))), 2, (255,255,0), -1)
            cv2.line(overlay, (int(round(ex)),int(round(ey))), (int(round(cx)),int(round(cy))), (255,0,255), 1)

    for i, (x, y) in enumerate(centers):
        color = (0,255,255) if i == cidx else (0,0,255)
        draw_cross(overlay, x, y, color, size=12, thickness=1)

    cv2.imwrite(str(outdir / "grid_rings_crosshairs.png"), overlay)
    oh, ow = overlay.shape[:2]
    scale = 1024.0 / float(max(oh, ow))
    prev = cv2.resize(overlay, (int(ow*scale), int(oh*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(outdir / "preview_small.jpg"), prev, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # also save rotated BMP for MCP (optional convenience)
    bmp8 = cv2.cvtColor(gray_rot, cv2.COLOR_GRAY2BGR)
    out_bmp = outdir / "rotated_for_MCP.bmp"
    cv2.imwrite(str(out_bmp), bmp8)
    print(f"[INFO] Save for MCP: {out_bmp}")
    print(f"[INFO] X/Y Gain ≈ {1.0/px_per_mm:.6f} mm/px  (DPI ≈ {px_per_mm*25.4:.1f})")
    print(f"[INFO] Grid step (X=Y): {cell_mm} mm  | count: {rows}x{cols}")

    # CSVs (row-major)
    det_ordered = centers[order]
    pd.DataFrame({"row": rm_rows[order], "col": rm_cols[order], "x": det_ordered[:,0], "y": det_ordered[:,1]})\
        .to_csv(outdir / "centers.csv", index=False)

    rows_ell = []
    for e in outer_ellipses:
        if e is None: rows_ell.append([np.nan]*5)
        else:
            (ex, ey), (MA, ma), ang = e
            rows_ell.append([ex, ey, MA, ma, ang])
    pd.DataFrame(rows_ell, columns=["cx","cy","major","minor","angle_deg"]).to_csv(outdir / "ellipses.csv", index=False)

    pd.DataFrame({
        "row": rm_rows[order], "col": rm_cols[order],
        "det_x": centers[order,0], "det_y": centers[order,1],
        "ideal_x": ideal_x[order], "ideal_y": ideal_y[order],
        "err_x_mm": err_x_mm[order], "err_y_mm": err_y_mm[order], "err_norm_mm": err_norm_mm[order],
        "err_x_px": err_x_px[order], "err_y_px": err_y_px[order], "err_norm_px": err_norm_px[order],
    }).to_csv(outdir / "errors.csv", index=False)

    cfg = {
        "image_filename": str(image_path.name),
        "physical_size_mm": physical_size_mm,
        "ring_diam_mm": ring_diam_mm,
        "cell_mm": cell_mm,
        "rows": rows, "cols": cols, "expected_count": expected_count,
        "px_per_mm": float(px_per_mm),
        "center_mode": center_mode,
        "debug_diff": debug_diff,
        "rotation_deg": float(rotate_deg),
        "notes": "Rotation-only pipeline (no homography / no poly2 / no TPS)."
    }
    outdir.joinpath("run_config.json").write_text(json.dumps(cfg, indent=2))

    # ---- Optional: update a .cal file ----
    if prompt_bool("Update a .cal file using these deviations?", True):
        cal_name = prompt_str("Path to existing .cal (will not be overwritten)")
        cal_in = Path(cal_name).expanduser()
        if not cal_in.exists():
            print(f"[ERROR] .cal not found: {cal_in}")
        else:
            tol_mm      = prompt_float("Tolerance (mm) – values |Δ| < tol are set to 0", 0.001)
            comp_sign   = prompt_bool("Apply negative of measured error? (recommended)", True)
            mode        = prompt_str("Write mode ('replace' or 'increment')", "replace").lower()
            if mode not in ("replace","increment"):
                mode = "replace"

            # Make row×col arrays for Δx/Δy (mm)
            ex_mm = np.zeros((rows, cols), np.float64)
            ey_mm = np.zeros((rows, cols), np.float64)
            for r in range(rows):
                for c in range(cols):
                    # find index where rm_rows==r and rm_cols==c
                    idx = np.where((rm_rows == r) & (rm_cols == c))[0]
                    if len(idx):
                        i = idx[0]
                        ex_mm[r, c] = float(err_x_mm[i])
                        ey_mm[r, c] = float(err_y_mm[i])

            cal_out = outdir / f"updated_{Path(cal_in).name}"
            try:
                update_cal_from_errors(
                    cal_in=cal_in,
                    cal_out=cal_out,
                    rows=rows,
                    cols=cols,
                    err_table_x_mm=ex_mm,
                    err_table_y_mm=ey_mm,
                    tol_mm=tol_mm,
                    mode=mode,
                    compensate_sign=comp_sign,
                )
                print(f"[SUCCESS] Wrote: {cal_out}")
                print(f"          mode={mode}, tol={tol_mm} mm, compensate_sign={comp_sign}")
            except Exception as e:
                print("[ERROR] Updating .cal failed:", e)

    print("\n=== Done (rotation-only + CAL updater) ===")
    print("Saved to:", outdir)

if __name__ == "__main__":
    main()
