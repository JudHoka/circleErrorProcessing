from pathlib import Path
import cv2
import numpy as np


def find_center_circle(
    image_path: str = "IMG_9503.JPG", physical_size_mm: float = 148.0
) -> tuple[int, int, int, float, np.ndarray] | None:
    """Estimate the circle around the image midpoint by scanning for edges.

    The image is first cropped to a square using the smaller dimension so that
    the physical scale (in millimetres) can be derived from the provided
    ``physical_size_mm``. A small window around the midpoint is then searched
    for circles. The circle nearest that window's center is selected.

    Parameters
    ----------
    image_path: str
        Path to the image containing circles.
    physical_size_mm: float
        The physical size of the square image in millimetres. This allows the
        function to report the pixel-to-millimetre scale.

    Returns
    -------
    tuple[int, int, int, float, numpy.ndarray] | None
        (x, y, radius, mm_per_pixel, cropped_color_image) of the detected circle,
        or ``None`` if not found.
    """

    color = cv2.imread(image_path)
    if color is None:
        raise FileNotFoundError(f"{image_path} not found")

    # Crop to central square so physical scaling is consistent
    h, w = color.shape[:2]
    size = min(h, w)
    x_offset = (w - size) // 2
    y_offset = (h - size) // 2
    color = color[y_offset : y_offset + size, x_offset : x_offset + size]

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    cx0, cy0 = size // 2, size // 2

    # Focus on a small region around the image center to avoid picking edge circles
    roi_size = 200
    x1, y1 = cx0 - roi_size // 2, cy0 - roi_size // 2
    x2, y2 = x1 + roi_size, y1 + roi_size
    roi = gray[y1:y2, x1:x2]

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        roi_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=20,
        minRadius=5,
        maxRadius=50,
    )
    if circles is None:
        return None

    # Choose the circle closest to the ROI center
    circles = np.round(circles[0]).astype(int)
    roi_center = np.array([roi.shape[1] // 2, roi.shape[0] // 2])
    cx, cy, r = min(
        circles, key=lambda c: np.linalg.norm(c[:2] - roi_center)
    )
    cx += x1
    cy += y1
    radius = int(r)

    # Calculate pixel-to-mm scale for the cropped square
    mm_per_pixel = physical_size_mm / size

    return cx, cy, radius, mm_per_pixel, color


def draw_grid(
    img: np.ndarray,
    center: tuple[int, int],
    mm_per_pixel: float,
    cell_mm: float = 4.0,
    offset_mm: float = 2.0,
) -> None:
    """Overlay a square grid on ``img`` centred around ``center``.

    Grid cells are ``cell_mm`` wide, with the first horizontal and vertical
    lines offset ``offset_mm`` from ``center``.
    """

    h, w = img.shape[:2]
    cx, cy = center
    cell_px = int(round(cell_mm / mm_per_pixel))
    offset_px = int(round(offset_mm / mm_per_pixel))

    # Vertical lines to the left and right of the centre
    for x in range(cx - offset_px, -1, -cell_px):
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)
    for x in range(cx + offset_px, w, cell_px):
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)

    # Horizontal lines above and below the centre
    for y in range(cy - offset_px, -1, -cell_px):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    for y in range(cy + offset_px, h, cell_px):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)


def detect_all_circles(
    gray: np.ndarray, mm_per_pixel: float, cell_mm: float = 4.0
) -> np.ndarray | None:
    """Detect all circles in ``gray`` using Hough transform.

    Parameters
    ----------
    gray: numpy.ndarray
        Grayscale image containing the circle grid.
    mm_per_pixel: float
        Millimetres represented by a single pixel.
    cell_mm: float
        Expected spacing between circle centres.
    """

    cell_px = cell_mm / mm_per_pixel
    expected_r_px = (1.5) / mm_per_pixel  # 3 mm diameter -> 1.5 mm radius

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    circles = cv2.HoughCircles(
        opened,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(cell_px * 0.9),
        param1=150,
        param2=60,
        minRadius=int(expected_r_px * 0.9),
        maxRadius=int(expected_r_px * 1.1),
    )

    if circles is not None:
        return np.round(circles[0]).astype(int)
    return None


def compute_offsets(
    circles: np.ndarray,
    center: tuple[int, int],
    mm_per_pixel: float,
    cell_mm: float = 4.0,
) -> list[dict[str, float | str | int]]:
    """Compute positional offsets of circles from their ideal grid points."""

    cx, cy = center
    cell_px = cell_mm / mm_per_pixel

    results: list[dict[str, float | str | int]] = []
    for x, y, _ in circles:
        dx_mm = (x - cx) * mm_per_pixel
        dy_mm = (y - cy) * mm_per_pixel

        gx = round(dx_mm / cell_mm)
        gy = round(dy_mm / cell_mm)

        expected_x = cx + gx * cell_px
        expected_y = cy + gy * cell_px

        offset_x = (x - expected_x) * mm_per_pixel
        offset_y = (y - expected_y) * mm_per_pixel

        label = "CENTER" if gx == 0 and gy == 0 else f"{gx},{gy}"

        results.append(
            {
                "circle": label,
                "grid_x": gx,
                "grid_y": gy,
                "offset_x_mm": offset_x,
                "offset_y_mm": offset_y,
            }
        )

    return results

if __name__ == "__main__":
    result = find_center_circle()
    if result:
        x, y, r, mm_per_pixel, img = result
        print(f"Center circle: x={x}, y={y}, radius={r} pixels")
        print(
            "Center circle (mm): x={:.2f}, y={:.2f}, radius={:.2f}".format(
                x * mm_per_pixel, y * mm_per_pixel, r * mm_per_pixel
            )
        )
        
        gray_all = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = detect_all_circles(gray_all, mm_per_pixel)
        if circles is not None:
            offsets = compute_offsets(circles, (x, y), mm_per_pixel)

            try:
                import pandas as pd
            except ImportError:
                pd = None

            if pd is not None:
                df = pd.DataFrame(offsets)
                output_dir = Path.home() / "downloads"
                output_dir.mkdir(parents=True, exist_ok=True)
                excel_path = output_dir / "circle_offsets.xlsx"
                try:
                    df.to_excel(excel_path, index=False)
                    print(f"Saved offsets to {excel_path}")
                except ImportError:
                    csv_path = output_dir / "circle_offsets.csv"
                    df.to_csv(csv_path, index=False)
                    print(
                        f"openpyxl not installed; saved offsets to {csv_path}"
                    )
            else:
                print("pandas not installed; skipping Excel export")
                
        draw_grid(img, (x, y), mm_per_pixel)
        output_dir = Path.home() / "downloads"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "IMG_9503_grid.png"
        cv2.imwrite(str(output_path), img)
        print(f"Saved grid image to {output_path}")
    else:
        print("No circles detected")

