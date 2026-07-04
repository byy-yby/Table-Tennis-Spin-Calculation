"""球边缘子像素精炼: 径向梯度扫描 + RANSAC 圆拟合。

替代 BallNet/Hough 的几何方法。核心思想: 单点边缘受锯齿限制(~1px),
但圆拟合聚合 N 个边缘点, 中心精度 ≈ 单点噪声/√N, 50 点可达 ~0.2px。
RANSAC 自动排除黑点内部边缘/遮挡弧。
"""
import cv2, numpy as np


def _radial_edge_points(img, cx, cy, r_rough,
                        n_rays=72, r_frac=(0.5, 1.5)):
    """
    沿 n_rays 条径向射线扫描 Sobel 梯度峰值, 返回子像素边缘点列表。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Sobel 梯度幅值
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)

    r_min = max(2.0, r_rough * r_frac[0])
    r_max = min(min(h, w) / 2.0 - 1, r_rough * r_frac[1])
    if r_max <= r_min:
        return []

    pts = []
    for k in range(n_rays):
        theta = 2 * np.pi * k / n_rays
        dx, dy = np.cos(theta), np.sin(theta)
        # 沿射线采样梯度
        rs = np.arange(r_min, r_max, 0.5)
        xs = cx + rs * dx
        ys = cy + rs * dy
        xi = xs.astype(int)
        yi = ys.astype(int)
        mask = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if np.sum(mask) < 3:
            continue
        rs_m, xi_m, yi_m = rs[mask], xi[mask], yi[mask]
        vals = gmag[yi_m, xi_m]
        # 球边缘 = 最外层"显著"梯度峰。显著 = > 25% 该射线最大梯度
        # (排除背景噪声峰 ~5 vs 球边缘 ~50; 也避开内部黑点边缘 — 球边缘是显著峰中最外层)
        i_max = _outermost_peak(vals, frac=0.25)
        if i_max is None:
            continue
        # 抛物线子像素插值 (3 点)
        if 0 < i_max < len(vals) - 1:
            y0, y1, y2 = vals[i_max - 1], vals[i_max], vals[i_max + 1]
            denom = (y0 - 2 * y1 + y2)
            offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            offset = np.clip(offset, -0.5, 0.5)
            r_star = rs_m[i_max] + offset * 0.5
        else:
            r_star = rs_m[i_max]
        pts.append((cx + r_star * dx, cy + r_star * dy))
    return pts


def _outermost_peak(vals, frac=0.25):
    """
    返回最外层显著局部极大值索引。
    显著 = vals[i] > frac * max(vals)。无显著峰则返回 None。
    """
    n = len(vals)
    if n == 0:
        return None
    thresh = frac * np.max(vals)
    peaks = [i for i in range(1, n - 1)
             if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1] and vals[i] > thresh]
    if peaks:
        return peaks[-1]            # 最外层显著峰
    return None


def _ransac_circle(pts, r_rough, max_iter=80, tol=1.5):
    """
    RANSAC 圆拟合 (3 点采样, 代数解)。返回 (cx, cy, r, inlier_mask)。
    """
    n = len(pts)
    if n < 3:
        return None
    pts = np.array(pts, dtype=np.float64)
    best_inliers = 0
    best = None

    for _ in range(max_iter):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx]
        c, r = _circle_from_3pts(p1, p2, p3)
        if c is None or not (r_rough * 0.7 < r < r_rough * 1.3):
            continue
        # 内点: 点到圆周距离 < tol
        d = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - r)
        inliers = d < tol
        n_in = np.sum(inliers)
        if n_in > best_inliers:
            best_inliers = n_in
            best = (c, r, inliers)

    if best is None or best_inliers < 3:
        return None

    # 全体内点最小二乘精炼 (Kåsa 代数法)
    c, r, inliers = best
    pin = pts[inliers]
    A = np.c_[2 * pin[:, 0], 2 * pin[:, 1], np.ones(len(pin))]
    b = pin[:, 0] ** 2 + pin[:, 1] ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx_f, cy_f = sol[0], sol[1]
        r_f = np.sqrt(max(sol[2] + cx_f ** 2 + cy_f ** 2, 1e-9))
        if r_rough * 0.7 < r_f < r_rough * 1.3:
            return (cx_f, cy_f, r_f)
    except Exception:
        pass
    return (c[0], c[1], r)


def _circle_from_3pts(p1, p2, p3):
    """三点定圆。返回 ((cx,cy), r) 或 (None, 0)。"""
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, 0.0
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    return (ux, uy), r


def refine_ball_edge(img, cx, cy, r, n_rays=72):
    """
    子像素球边缘精炼。输入粗 (cx,cy,r), 返回精炼 (cx,cy,r) 或 None(失败)。
    """
    pts = _radial_edge_points(img, cx, cy, r, n_rays=n_rays)
    if len(pts) < 6:
        return None
    result = _ransac_circle(pts, r)
    if result is None:
        return None
    cx_f, cy_f, r_f = result
    # 合理性检查: 偏移不超过粗半径的 30%
    if np.hypot(cx_f - cx, cy_f - cy) > 0.3 * r:
        return None
    return float(cx_f), float(cy_f), float(r_f)


def detect_ball(img, rough=None):
    """
    球检测入口。rough=(cx,cy,r) 可选粗估计 (来自 BG 减除/BallNet)。
    无 rough 时用图像中心 + 短边的 42% 作粗半径。
    返回 (cx, cy, r) 或 None。
    """
    h, w = img.shape[:2]
    if rough is None:
        cx, cy, r = w / 2.0, h / 2.0, min(w, h) * 0.33
    else:
        cx, cy, r = rough
    refined = refine_ball_edge(img, cx, cy, r)
    if refined is not None:
        return refined
    # 回退: 返回粗估计
    return (float(cx), float(cy), float(r))
