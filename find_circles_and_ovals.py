from pathlib import Path
import cv2
import numpy as np

from find_center_circle import find_center_circle, draw_grid, detect_all_circles


def draw_crosshair(
    img: np.ndarray,
    center: tuple[int, int],
    size: int = 5,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw a simple crosshair centered at ``center`` on ``img``."""

    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, 1)
    cv2.line(img, (x, y - size), (x, y + size), color, 1)


def detect_ellipses(
    gray: np.ndarray, mm_per_pixel: float, cell_mm: float = 4.0
) -> np.ndarray | None:
    """Detect ovals (ellipses) in ``gray`` using contour fitting."""

    expected_r_px = 1.5 / mm_per_pixel
    expected_d_px = expected_r_px * 2

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ellipses: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse

        if not (
            expected_d_px * 0.75 <= ma <= expected_d_px * 1.25
            and expected_d_px * 0.75 <= MA <= expected_d_px * 1.35
        ):
            continue
        if abs(MA - ma) < 3:  # nearly circular; handled elsewhere
            continue

        ellipses.append(
            (
                int(round(x)),
                int(round(y)),
                int(round(MA / 2)),
                int(round(ma / 2)),
                angle,
            )
        )

    if ellipses:
        return np.array(ellipses, dtype=float)
    return None


def process_image(
    image_path: str = "IMG_9503.JPG",
) -> tuple[np.ndarray | None, np.ndarray | None, Path]:
    """Detect circles and ovals, overlay a grid and save an annotated image."""

    result = find_center_circle(image_path)
    if result is None:
        raise RuntimeError("No center circle detected")

    cx, cy, r, mm_per_pixel, img = result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = detect_all_circles(gray, mm_per_pixel)
    ellipses = detect_ellipses(gray, mm_per_pixel)

    # Draw the grid first so overlays remain visible
    draw_grid(img, (cx, cy), mm_per_pixel)

    if circles is not None:
        for x, y, rad in circles:
            cv2.circle(img, (x, y), rad, (0, 0, 255), 1)
            draw_crosshair(img, (x, y))

    if ellipses is not None:
        for x, y, major, minor, angle in ellipses:
            cv2.ellipse(img, (int(x), int(y)), (int(major), int(minor)), angle, 0, 360, (255, 0, 0), 1)
            draw_crosshair(img, (int(x), int(y)))

    # Highlight centre circle on top of the grid
    cv2.circle(img, (cx, cy), r, (0, 0, 255), 1)
    draw_crosshair(img, (cx, cy))

    output_dir = Path.home() / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "IMG_9503_shapes.png"
    cv2.imwrite(str(output_path), img)

    return circles, ellipses, output_path


if __name__ == "__main__":
    circles, ellipses, output_path = process_image()
    print(f"Detected {0 if circles is None else len(circles)} circles")
    print(f"Detected {0 if ellipses is None else len(ellipses)} ovals")
    print(f"Saved annotated image to {output_path}")
