"""单帧处理: 霍夫圆球检测 + CNN黑点 + 2D→3D"""
import cv2, torch, numpy as np
from scipy.ndimage import maximum_filter
from .config import (HOUGH_DP, HOUGH_PARAM1, HOUGH_PARAM2, CNN_HMAP_THRESH,
                      BALL_RADIUS_MM, DEVICE)
from .geometry import ball_center_to_3d, ray_sphere_intersection


def process_image(img, model, device=None):
    """处理单张图: 霍夫圆 → CNN黑点 → 2D→3D"""
    if device is None: device = DEVICE
    h, w = img.shape[:2]

    # 霍夫圆: 4x 放大
    scale = 4
    img_big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray_big = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_big, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=50,
                                param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                                minRadius=30, maxRadius=150)
    if circles is None:
        return [], [], None

    bcx = float(circles[0][0][0]) / scale
    bcy = float(circles[0][0][1]) / scale
    br = float(circles[0][0][2]) / scale

    # CNN 黑点检测
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()
    peaks = (pred == maximum_filter(pred, size=5)) & (pred > CNN_HMAP_THRESH)
    ys, xs = np.where(peaks)
    dots_2d = list(zip(xs, ys))

    # 2D→3D
    bc_3d = ball_center_to_3d(bcx, bcy, br)
    dots_3d = []
    for dx, dy in dots_2d:
        p3d = ray_sphere_intersection(dx, dy, bc_3d)
        if p3d is not None:
            v = (p3d - bc_3d) / BALL_RADIUS_MM
            dots_3d.append(np.array([v[0], v[1], v[2]]))

    return dots_2d, dots_3d, (bcx, bcy, br, bc_3d)
