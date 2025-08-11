import cv2
import numpy as np

# Load image
img = cv2.imread("IMG_9503.JPG")
if img is None:
    raise FileNotFoundError("IMG_9503.JPG not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=100,
    param2=30,
    minRadius=5,
    maxRadius=30,
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    h, w = gray.shape[:2]
    image_center = np.array([w // 2, h // 2])

    # Select circle whose center is closest to image center
    circle_center = min(
        circles, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - image_center)
    )
    x, y, r = circle_center
    print(f"Center circle: x={x}, y={y}, radius={r}")
else:
    print("No circles detected")
