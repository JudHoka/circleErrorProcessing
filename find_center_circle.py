import cv2
import numpy as np


def find_center_circle(
    image_path: str = "IMG_9503.JPG", physical_size_mm: float = 148.0
) -> tuple[int, int, int, float] | None:
    """Estimate the circle around the image midpoint by scanning for edges.

    The image is first cropped to a square using the smaller dimension so that
    the physical scale (in millimetres) can be derived from the provided
    ``physical_size_mm``. A small window around the midpoint is then searched
    for circles. The circle nearest that window's center is selected, and its
    radius is refined by scanning outward from the detected center until a
    strong intensity change (edge) is encountered.

    Parameters
    ----------
    image_path: str
        Path to the image containing circles.
    physical_size_mm: float
        The physical size of the square image in millimetres. This allows the
        function to report the pixel-to-millimetre scale.

    Returns
    -------
    tuple[int, int, int, float] | None
        (x, y, radius, mm_per_pixel) of the detected circle, or ``None`` if not
        found.
    """

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")

    # Crop to central square so physical scaling is consistent
    h, w = img.shape
    size = min(h, w)
    x_offset = (w - size) // 2
    y_offset = (h - size) // 2
    img = img[y_offset : y_offset + size, x_offset : x_offset + size]
    h = w = size

    cx0, cy0 = w // 2, h // 2

    # Focus on a small region around the image center to avoid picking edge circles
    roi_size = 200
    x1, y1 = cx0 - roi_size // 2, cy0 - roi_size // 2
    x2, y2 = x1 + roi_size, y1 + roi_size
    roi = img[y1:y2, x1:x2]

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

    return cx, cy, radius, mm_per_pixel

if __name__ == "__main__":
    result = find_center_circle()
    if result:
        x, y, r, mm_per_pixel = result
        print(f"Center circle: x={x}, y={y}, radius={r} pixels")
        print(
            "Center circle (mm): x={:.2f}, y={:.2f}, radius={:.2f}".format(
                x * mm_per_pixel, y * mm_per_pixel, r * mm_per_pixel
            )
        )
    else:
        print("No circles detected")

