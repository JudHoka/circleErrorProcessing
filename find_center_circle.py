import cv2
import numpy as np


def find_center_circle(image_path: str = "IMG_9503.JPG") -> tuple[int, int, int] | None:
    """Estimate the circle around the image midpoint by scanning for edges.

    The function crops a small window around the image midpoint and searches it
    for circles. The circle nearest the window's center is selected, and its
    radius is refined by scanning outward from the detected center until a
    strong intensity change (edge) is encountered.

    Parameters
    ----------
    image_path: str
        Path to the image containing circles.

    Returns
    -------
    tuple[int, int, int] | None
        (x, y, radius) of the detected circle, or ``None`` if not found.
    """

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")

    h, w = img.shape
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
    cx, cy, _ = min(
        circles, key=lambda c: np.linalg.norm(c[:2] - roi_center)
    )
    cx += x1
    cy += y1

    # Use intensity scanning from the detected center to refine the radius
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    center_val = int(blurred[cy, cx])

    def scan_from_center(dx: int, dy: int, threshold: int = 40) -> int:
        x, y, dist = cx, cy, 0
        while 0 <= x + dx < w and 0 <= y + dy < h:
            x += dx
            y += dy
            dist += 1
            if abs(int(blurred[y, x]) - center_val) > threshold:
                break
        return dist

    left = scan_from_center(-1, 0)
    right = scan_from_center(1, 0)
    up = scan_from_center(0, -1)
    down = scan_from_center(0, 1)

    if min(left, right, up, down) == 0:
        return None

    radius = int(round((left + right + up + down) / 4))
    return cx, cy, radius


if __name__ == "__main__":
    result = find_center_circle()
    if result:
        x, y, r = result
        print(f"Center circle: x={x}, y={y}, radius={r}")
    else:
        print("No circles detected")

