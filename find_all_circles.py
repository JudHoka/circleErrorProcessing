from pathlib import Path
import cv2
import numpy as np

from find_center_circle import (
    find_center_circle,
    draw_grid,
    detect_all_circles,
)


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


def process_image(image_path: str = "04_unsharp.png") -> tuple[np.ndarray, Path]:
    """Detect circles, overlay a grid and save an annotated image."""

    result = find_center_circle(image_path)
    if result is None:
        raise RuntimeError("No center circle detected")

    cx, cy, r, mm_per_pixel, img = result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = detect_all_circles(gray, mm_per_pixel)

    # Draw the grid first so circle overlays remain visible
    draw_grid(img, (cx, cy), mm_per_pixel)

    if circles is None:
        circles = np.empty((0, 3), dtype=int)
    else:
        for x, y, rad in circles:
            cv2.circle(img, (x, y), rad, (0, 0, 255), 1)
            draw_crosshair(img, (x, y))

    # Ensure the detected centre circle is highlighted on top of the grid
    cv2.circle(img, (cx, cy), r, (0, 0, 255), 1)
    draw_crosshair(img, (cx, cy))

    output_dir = Path.home() / "downloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "IMG_9503_grid.png"
    cv2.imwrite(str(output_path), img)

    return circles, output_path


if __name__ == "__main__":
    circles, output_path = process_image()
    print(f"Detected {len(circles)} circles")
    print(f"Saved annotated image to {output_path}")
