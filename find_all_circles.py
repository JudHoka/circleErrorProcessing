from pathlib import Path
import cv2
import numpy as np


def draw_crosshair(img: np.ndarray, center: tuple[int, int], size: int = 5, color: tuple[int, int, int] = (0, 0, 255)) -> None:
    """Draw a simple crosshair centered at ``center`` on ``img``.

    Parameters
    ----------
    img: numpy.ndarray
        Image on which to draw.
    center: tuple[int, int]
        (x, y) pixel location of the crosshair center.
    size: int
        Half-length of each crosshair arm in pixels.
    color: tuple[int, int, int]
        BGR colour for the crosshair.
    """
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, 1)
    cv2.line(img, (x, y - size), (x, y + size), color, 1)


def find_all_circles(image_path: str = "IMG_9503.JPG") -> tuple[np.ndarray, Path]:
    """Detect all circles in ``image_path`` and save an annotated image.

    Returns
    -------
    tuple[np.ndarray, Path]
        Array of detected circles (x, y, r) and path to the output image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=5,
        maxRadius=80,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            draw_crosshair(img, (x, y))
            cv2.circle(img, (x, y), r, (0, 255, 0), 1)
    else:
        circles = np.empty((0, 3), dtype=int)

    output_dir = Path.home() / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "IMG_9503_circles.png"
    cv2.imwrite(str(output_path), img)

    return circles, output_path


if __name__ == "__main__":
    circles, output_path = find_all_circles()
    print(f"Detected {len(circles)} circles")
    print(f"Saved annotated image to {output_path}")
