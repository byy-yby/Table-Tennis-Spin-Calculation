"""几何工具: 3D球心, 射线求交, Kabsch"""
import cv2, numpy as np
from .config import MTX, DIST, BALL_RADIUS_MM


def ball_center_to_3d(cx, cy, r_px, mtx=None, ball_radius_mm=None):
    if mtx is None: mtx = MTX
    if ball_radius_mm is None: ball_radius_mm = BALL_RADIUS_MM
    fx, fy = mtx[0, 0], mtx[1, 1]
    Z = (fx * ball_radius_mm) / r_px
    X = (cx - mtx[0, 2]) * Z / fx
    Y = (cy - mtx[1, 2]) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)


def ray_sphere_intersection(u, v, center_3d, mtx=None, dist=None, ball_radius_mm=None):
    if mtx is None: mtx = MTX
    if dist is None: dist = DIST
    if ball_radius_mm is None: ball_radius_mm = BALL_RADIUS_MM
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    npt = cv2.undistortPoints(pt, mtx, dist)[0][0]
    d = np.array([npt[0], npt[1], 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    b = -2.0 * np.dot(d, center_3d)
    c = np.dot(center_3d, center_3d) - ball_radius_mm ** 2
    t = (-b - np.sqrt(max(b ** 2 - 4 * c, 0))) / 2.0
    return t * d


def kabsch(ps, pd):
    """不减质心版 Kabsch (点已是球心相对向量)"""
    H = ps.T @ pd
    U, S, Vh = np.linalg.svd(H)
    det = np.sign(np.linalg.det(Vh.T @ U.T))
    R = Vh.T @ np.diag([1., 1., det]) @ U.T
    cos_t = np.clip((np.trace(R) - 1.) / 2., -1., 1.)
    theta = np.arccos(cos_t)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    nrm = np.linalg.norm(axis)
    axis = axis / nrm if nrm > 1e-9 else np.array([0., 0., 1.])
    return R, theta, axis
