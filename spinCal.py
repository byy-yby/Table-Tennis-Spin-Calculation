#!/usr/bin/env python3
# ============================================================
# spinCal.py — 手动标注黑点 + 几何法解算乒乓球旋转
#
# 交互：
#   左键点击  = 添加点（自动 ID）
#   数字键0-9 = 指定下一个点的 ID
#   右键      = 删除上一个点
#   ENTER     = 保存当前帧，前进到下一帧
#   Backspace = 退回上一帧
#   F         = 完成标注，计算 RPM + 旋转轴
#   Q / ESC   = 退出
#
# 用法:
#   python spinCal.py
#   python spinCal.py D:\highspeed\test5.avi
# ============================================================

import cv2
import numpy as np
import os
import sys

# ── 相机参数 ──
MTX = np.array([
    [1914.91362, 0.0,       802.626903],
    [0.0,       1906.44992, 542.879886],
    [0.0,       0.0,        1.0]
], dtype=np.float64)
DIST = np.array([-0.17027915, 0.220013, 0.00154302, -0.00201744, 1.28603106], dtype=np.float64)
BALL_RADIUS_MM = 20.0
CALIB_IMAGE_HEIGHT = 1080


def adjust_principal_point(mtx, video_h):
    if video_h == CALIB_IMAGE_HEIGHT:
        return mtx.copy()
    crop = (CALIB_IMAGE_HEIGHT - video_h) / 2.0
    m2 = mtx.copy()
    m2[1, 2] -= crop
    print(f"[Calib] cy: {mtx[1,2]:.1f} -> {m2[1,2]:.1f}")
    return m2


def ball_center_to_3d(cx, cy, r_px, mtx):
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx0, cy0 = mtx[0, 2], mtx[1, 2]
    Z = (fx * BALL_RADIUS_MM) / r_px
    X = (cx - cx0) * Z / fx
    Y = (cy - cy0) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)


def ray_sphere_intersection(u, v, center_3d, radius_mm, mtx, dist):
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    npt = cv2.undistortPoints(pt, mtx, dist)[0][0]
    d = np.array([npt[0], npt[1], 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    b = -2.0 * np.dot(d, center_3d)
    c = np.dot(center_3d, center_3d) - radius_mm**2
    delta = max(b**2 - 4*c, 0)
    t = (-b - np.sqrt(delta)) / 2.0
    return t * d


def kabsch(pts_src, pts_dst):
    c1 = np.mean(pts_src, axis=0)
    c2 = np.mean(pts_dst, axis=0)
    H = (pts_src - c1).T @ (pts_dst - c2)
    U, S, Vh = np.linalg.svd(H)
    det = np.sign(np.linalg.det(Vh.T @ U.T))
    R = Vh.T @ np.diag([1.0, 1.0, det]) @ U.T
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    nrm = np.linalg.norm(axis)
    axis = axis / nrm if nrm > 1e-9 else np.array([0., 0., 1.])
    return R, theta, axis


def to_user_axis(axis_cam):
    return np.array([axis_cam[1], axis_cam[0], -axis_cam[2]])


# ═══════════════════════════════════════════════════════════
#  球窗口扫描 (与之前一样)
# ═══════════════════════════════════════════════════════════

def build_background(cap, total_frames):
    samples = []
    for fi in np.linspace(0, total_frames-1, min(12, total_frames), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.median(np.array(samples), axis=0)


def detect_ball_hough(gray, cx_hint, cy_hint, r_hint):
    best, best_score = None, -1e9
    h, w = gray.shape
    r_min, r_max = max(15, int(r_hint*0.7)), min(150, int(r_hint*1.3))
    for dp in [1.1, 1.2]:
        for p2 in [20, 25, 30, 35]:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=50,
                param1=40, param2=p2, minRadius=r_min, maxRadius=r_max
            )
            if circles is None: continue
            for (cx, cy, r) in np.uint16(np.around(circles))[0]:
                # 跳过与提示位置差异过大的圆
                if abs(cx-cx_hint) > r_hint*0.5 or abs(cy-cy_hint) > r_hint*0.5:
                    continue
                # 只接受完整可见的圆（确保圆完全在图像内）
                if cx - r < 1 or cy - r < 1 or cx + r >= w-1 or cy + r >= h-1:
                    continue
                s = _score_circle(gray, cx, cy, r)
                if s > best_score:
                    best_score = s
                    best = (float(cx), float(cy), float(r), s)
    return best


def _score_circle(gray, cx, cy, r):
    h, w = gray.shape
    n, grads, inners, outers = 24, [], [], []
    for a in np.linspace(0, 2*np.pi, n, endpoint=False):
        ex, ey = int(cx+r*np.cos(a)), int(cy+r*np.sin(a))
        if 1 <= ex < w-1 and 1 <= ey < h-1:
            grads.append(abs(int(gray[ey,ex+1])-int(gray[ey,ex-1])) +
                         abs(int(gray[ey+1,ex])-int(gray[ey-1,ex])))
        ix, iy = int(cx+0.7*r*np.cos(a)), int(cy+0.7*r*np.sin(a))
        if 0 <= ix < w and 0 <= iy < h: inners.append(gray[iy,ix])
        ox, oy = int(cx+1.2*r*np.cos(a)), int(cy+1.2*r*np.sin(a))
        if 0 <= ox < w and 0 <= oy < h: outers.append(gray[oy,ox])
    if not grads or not inners or not outers: return -1e9
    return np.mean(grads) + 2.0*max(0, np.mean(inners)-np.mean(outers))


def binarize_ball_region(gray, cx, cy, r, out_size=150, thresh=0):
    """对霍夫圆内区域做掩码并二值化，返回 BGR 显示图像（out_size x out_size）。

    参数:
      thresh: 0 表示使用 Otsu，自定义阈值范围 1-255 表示固定阈值
    """
    h, w = gray.shape
    x0 = max(0, int(np.floor(cx - r)))
    y0 = max(0, int(np.floor(cy - r)))
    x1 = min(w, int(np.ceil(cx + r)) + 1)
    y1 = min(h, int(np.ceil(cy + r)) + 1)
    roi = gray[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    mask = np.zeros_like(roi, dtype=np.uint8)
    cx_local = int(round(cx)) - x0
    cy_local = int(round(cy)) - y0
    rr = int(round(r))
    cv2.circle(mask, (cx_local, cy_local), rr, 255, -1)

    # 应用掩码后二值化（只影响圆内像素）
    masked = cv2.bitwise_and(roi, roi, mask=mask)
    if np.count_nonzero(mask) > 0:
        if thresh and 1 <= thresh <= 255:
            _, bin_roi = cv2.threshold(masked, int(thresh), 255, cv2.THRESH_BINARY)
        else:
            _, bin_roi = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 保留圆外为黑
        bin_roi[mask == 0] = 0
    else:
        bin_roi = masked

    # 转为 BGR 并缩放到指定大小
    bin_bgr = cv2.cvtColor(bin_roi, cv2.COLOR_GRAY2BGR)
    bin_resized = cv2.resize(bin_bgr, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # 在二值图上标记中心点（黄色）以示意1像素标记位置
    cv2.circle(bin_resized, (out_size//2, out_size//2), 1, (0, 255, 255), -1)
    return bin_resized


def find_ball_window(cap, total_frames):
    print("\n[1/2] Scanning for ball window...")
    bg = build_background(cap, total_frames)

    print("  Phase 1: motion...")
    motion_frames = set()
    for fi in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - bg)
        _, mask = cv2.threshold(diff.astype(np.uint8), 2, 255, cv2.THRESH_BINARY)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        max_area = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)), default=0)
        if max_area > 500: motion_frames.add(fi)
        if fi % 400 == 0: print(f"    {fi}/{total_frames}...")

    if not motion_frames:
        for fi in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = np.abs(gray.astype(np.float32) - bg)
            _, mask = cv2.threshold(diff.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            max_area = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)), default=0)
            if max_area > 100: motion_frames.add(fi)
        if not motion_frames: raise RuntimeError("No motion")

    frames = sorted(motion_frames)
    best_start, best_end, best_len = frames[0], frames[0], 0
    cur = frames[0]
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] > 5:
            length = frames[i-1] - cur + 1
            if length > best_len: best_len, best_start, best_end = length, cur, frames[i-1]
            cur = frames[i]
    if frames[-1] - cur + 1 > best_len: best_start, best_end = cur, frames[-1]
    best_start, best_end = max(0, best_start-3), min(total_frames-1, best_end+3)
    print(f"  Motion: {best_start} ~ {best_end}")

    print("  Phase 2: Hough...")
    detections = {}
    for fi in range(best_start, best_end+1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - bg)
        _, mask = cv2.threshold(diff.astype(np.uint8), 2, 255, cv2.THRESH_BINARY)
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best_comp = max(
            ((stats[i, cv2.CC_STAT_AREA], centroids[i], stats[i]) for i in range(1, n_labels)),
            key=lambda x: x[0], default=None
        )
        if best_comp is None or best_comp[0] < 80: continue
        bcx, bcy = best_comp[1]
        br = max(best_comp[2][cv2.CC_STAT_WIDTH], best_comp[2][cv2.CC_STAT_HEIGHT])/2.0
        ball = detect_ball_hough(gray, bcx, bcy, br)
        if ball and ball[2] >= 15:
            detections[fi] = (ball[0], ball[1], ball[2])

    if len(detections) < 3:
        raise RuntimeError(f"Only {len(detections)} frames with ball")
    dk = sorted(detections.keys())
    print(f"  Detected: {len(detections)} frames, {dk[0]} ~ {dk[-1]}")
    return dk[0], dk[-1], detections


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════

DOT_COLORS = [
    (0, 255, 0),      # ID 0: green
    (0, 255, 255),    # ID 1: yellow
    (255, 128, 0),    # ID 2: orange
    (255, 0, 255),    # ID 3: magenta
    (0, 165, 255),    # ID 4: light blue
    (255, 255, 0),    # ID 5: cyan-yellow
    (128, 255, 0),    # ID 6: lime
    (255, 0, 128),    # ID 7: pink
    (0, 255, 128),    # ID 8: spring green
    (128, 128, 255),  # ID 9: lavender
]


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\highspeed\test5.avi"
    if not os.path.exists(video_path):
        alt = r"c:\Users\z\Desktop\BDprogram\highspeed\test5.avi"
        if os.path.exists(alt): video_path = alt
        else:
            print(f"Video not found: {video_path}"); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}"); return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Video: {video_w}x{video_h}, {fps:.1f} fps, {total_frames} frames")

    mtx = adjust_principal_point(MTX, video_h)

    # ── [1] 找球窗口 ──
    start_fr, end_fr, detections = find_ball_window(cap, total_frames)
    window_frames = sorted(detections.keys())
    median_r = np.median([detections[f][2] for f in window_frames])

    print(f"\n[2/2] Manual annotation — {len(window_frames)} frames to label")
    print(f"  LEFT CLICK  = add dot (auto ID)")
    print(f"  0-9         = set next dot ID")
    print(f"  RIGHT CLICK = remove last dot")
    print(f"  ENTER       = save frame & advance")
    print(f"  BACKSPACE   = go back one frame")
    print(f"  F           = finish & compute")
    print(f"  Q / ESC     = quit")

    # ── 标注状态 ──
    # all_annotations[frame_idx] = {dot_id: (px, py)}
    all_annotations = {}
    current_fi_idx = 0          # 当前在 window_frames 中的索引
    current_dots = {}           # 当前帧的点 {id: (px, py)}
    next_dot_id = 0             # 下一个自动分配的 ID
    manual_id = None            # 用户手动指定的 ID

    # 如果有之前标过的帧，加载它
    def load_frame(idx):
        nonlocal current_dots, next_dot_id
        fi = window_frames[idx]
        if fi in all_annotations:
            current_dots = dict(all_annotations[fi])
            next_dot_id = max(current_dots.keys()) + 1 if current_dots else 0
        else:
            current_dots = {}
            next_dot_id = 0

    load_frame(0)

    def mouse_cb(event, x, y, flags, param):
        nonlocal current_dots, next_dot_id, manual_id
        if event == cv2.EVENT_LBUTTONDOWN:
            dot_id = manual_id if manual_id is not None else next_dot_id
            # 如果这个 ID 已存在，覆盖
            current_dots[dot_id] = (float(x), float(y))
            if manual_id is not None:
                manual_id = None  # 用完即弃
            else:
                # 找到下一个未使用的 ID
                while next_dot_id in current_dots:
                    next_dot_id += 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_dots:
                # 删除最后添加的点
                last_id = max(current_dots.keys())
                del current_dots[last_id]
                next_dot_id = min(next_dot_id, last_id)

    win = "Manual Dot Annotation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    disp_w = min(1600, video_w)
    disp_h = int(disp_w * video_h / video_w)
    cv2.resizeWindow(win, disp_w, disp_h + 40)
    cv2.setMouseCallback(win, mouse_cb)
    # 二值化阈值滑块（0 = Otsu, 1-255 = 固定阈值）
    cv2.createTrackbar("Thresh", win, 0, 255, lambda x: None)

    def draw_frame(fi, dots, prev_dots):
        """绘制当前帧：球框 + 当前点 + 上一帧的参考点"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: return np.zeros((video_h, video_w, 3), dtype=np.uint8)

        display = frame.copy()
        cx, cy, r = detections[fi]

        # 球框
        cv2.circle(display, (int(cx), int(cy)), int(r), (255, 80, 0), 2)

        # 在球心处画一个 1px 半径的标记
        cv2.circle(display, (int(cx), int(cy)), 1, (0, 255, 255), -1)

        # 上一帧的点（半透明参考）
        for dot_id, (px, py) in prev_dots.items():
            color = DOT_COLORS[dot_id % len(DOT_COLORS)]
            cv2.circle(display, (int(px), int(py)), 5, color, 1)
            cv2.putText(display, str(dot_id), (int(px)+6, int(py)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 当前帧的点（实心）
        for dot_id, (px, py) in dots.items():
            color = DOT_COLORS[dot_id % len(DOT_COLORS)]
            cv2.circle(display, (int(px), int(py)), 5, color, -1)
            cv2.circle(display, (int(px), int(py)), 7, color, 1)
            cv2.putText(display, str(dot_id), (int(px)+8, int(py)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 信息面板
        cx, cy, r = detections[fi]
        idx = window_frames.index(fi) if fi in window_frames else 0
        n_labeled = len([f for f in all_annotations if len(all_annotations[f]) > 0])
        lines = [
            f"Frame {fi} ({idx+1}/{len(window_frames)}) | r={r:.0f}px",
            f"Dots: {list(dots.keys())} | Next auto-ID: {next_dot_id}" +
            (f" | Manual ID: {manual_id}" if manual_id is not None else ""),
            f"Labeled frames: {n_labeled} | ENTER=next BACKSPACE=prev F=finish",
        ]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (700, len(lines)*22+15), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)
        for i, line in enumerate(lines):
            cv2.putText(display, line, (10, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 生成二值化面板并贴到右上角
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thr = cv2.getTrackbarPos("Thresh", win)
            bin_img = binarize_ball_region(gray, cx, cy, r, out_size=150, thresh=thr)
            bh, bw = bin_img.shape[:2]
            margin = 10
            x0 = max(0, display.shape[1] - bw - margin)
            y0 = margin
            display[y0:y0+bh, x0:x0+bw] = bin_img
            # 框出二值化面板
            cv2.rectangle(display, (x0-1, y0-1), (x0+bw+1, y0+bh+1), (200, 200, 200), 1)
        except Exception:
            pass

        return display

    # 主循环
    while True:
        fi = window_frames[current_fi_idx]
        # 前一帧的点（参考）
        prev_dots = {}
        if current_fi_idx > 0:
            prev_fi = window_frames[current_fi_idx - 1]
            prev_dots = all_annotations.get(prev_fi, {})

        display = draw_frame(fi, current_dots, prev_dots)
        cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # ENTER: 保存并前进
            all_annotations[fi] = dict(current_dots)
            if current_fi_idx < len(window_frames) - 1:
                current_fi_idx += 1
                load_frame(current_fi_idx)
            else:
                print("\nAll frames annotated. Computing...")
                break
        elif key == 8:  # BACKSPACE: 回退
            all_annotations[fi] = dict(current_dots)  # 先保存当前
            if current_fi_idx > 0:
                current_fi_idx -= 1
                load_frame(current_fi_idx)
        elif key == ord('f') or key == ord('F'):  # F: 完成
            all_annotations[fi] = dict(current_dots)  # 保存当前
            print("\nAnnotation complete. Computing...")
            break
        elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC/Q
            cv2.destroyAllWindows()
            cap.release()
            return
        elif ord('0') <= key <= ord('9'):  # 数字键指定 ID
            manual_id = key - ord('0')
        elif key == ord('c') or key == ord('C'):  # C: 清除当前帧
            current_dots = {}
            next_dot_id = 0

    cv2.destroyWindow(win)

    # ═════════════════════════════════════════════════════
    #  旋转解算
    # ═════════════════════════════════════════════════════

    print(f"\n{'='*55}")
    print(f"  Computing rotation from {len(all_annotations)} labeled frames...")

    # 收集每帧的 3D 点
    frame_3d = {}  # {fi: {dot_id: 3d_vector_local}}

    for fi in all_annotations:
        cx, cy, r = detections[fi]
        ball_3d = ball_center_to_3d(cx, cy, median_r, mtx)
        dots_3d = {}
        for dot_id, (px, py) in all_annotations[fi].items():
            p3d = ray_sphere_intersection(px, py, ball_3d, BALL_RADIUS_MM, mtx, DIST)
            if p3d is not None:
                dots_3d[dot_id] = p3d - ball_3d  # 球体局部坐标
        if dots_3d:
            frame_3d[fi] = dots_3d

    labeled_frames = sorted(frame_3d.keys())
    if len(labeled_frames) < 2:
        print("  ERROR: Need at least 2 frames with dots")
        cap.release()
        return

    # ── Kabsch 逐帧累积 ──
    # 只用 3+ 公共点的帧对（2 点时 Kabsch 对噪声极度敏感）
    MIN_COMMON = 3
    dt = 1.0 / fps
    cumulative_theta = 0.0
    ref_axis = None
    n_reliable = 0      # 3+ 公共点的帧对数
    n_fallback = 0      # 2 公共点的帧对数（不可靠）
    pair_details = []   # (from_fi, to_fi, signed_deg, n_common, reliable)

    prev_fi = labeled_frames[0]
    prev_dots = frame_3d[prev_fi]

    for fi in labeled_frames[1:]:
        cur_dots = frame_3d[fi]
        common_ids = sorted(set(prev_dots.keys()) & set(cur_dots.keys()))
        n_common = len(common_ids)

        use_this_pair = (n_common >= MIN_COMMON)
        is_reliable = use_this_pair

        if n_common < 2:
            prev_dots = cur_dots
            prev_fi = fi
            continue

        if n_common == 2 and n_reliable == 0:
            # 没有任何可靠对时才回退到 2 点
            use_this_pair = True
            is_reliable = False

        if use_this_pair:
            pts_src = np.array([prev_dots[i] for i in common_ids])
            pts_dst = np.array([cur_dots[i] for i in common_ids])
            try:
                R, step_theta, step_axis = kabsch(pts_src, pts_dst)
                if ref_axis is None:
                    ref_axis = step_axis
                # 符号处理
                signed_theta = step_theta
                if np.dot(ref_axis, step_axis) < 0:
                    signed_theta = -step_theta
                cumulative_theta += signed_theta
                if is_reliable:
                    n_reliable += 1
                else:
                    n_fallback += 1
                pair_details.append((prev_fi, fi, np.degrees(signed_theta),
                                     n_common, is_reliable))
            except Exception:
                pass

        prev_dots = cur_dots
        prev_fi = fi

    total_theta = abs(cumulative_theta)
    n_pairs = n_reliable + n_fallback
    rpm = rps = 0.0
    axis_user = np.array([0., 0., 1.])
    reliability_note = ""

    if n_reliable > 0:
        omega = total_theta / (n_reliable * dt)
        rps = omega / (2 * np.pi)
        rpm = rps * 60
        if ref_axis is not None:
            axis_cam = ref_axis * np.sign(cumulative_theta) if cumulative_theta != 0 else ref_axis
            axis_user = to_user_axis(axis_cam)
        reliability_note = f"(using {n_reliable} reliable pairs with 3+ dots)"
    elif n_fallback > 0:
        omega = total_theta / (n_fallback * dt)
        rps = omega / (2 * np.pi)
        rpm = rps * 60
        if ref_axis is not None:
            axis_cam = ref_axis * np.sign(cumulative_theta) if cumulative_theta != 0 else ref_axis
            axis_user = to_user_axis(axis_cam)
        reliability_note = "⚠️  FALLBACK: used 2-dot pairs only, axis may be unreliable"
    else:
        reliability_note = "No usable frame pairs"

    # 线速度
    velocities, vel_fis = [], []
    prev_center = None
    for fi in labeled_frames:
        cx, cy, r = detections[fi]
        center = ball_center_to_3d(cx, cy, median_r, mtx)
        if prev_center is not None:
            velocities.append(np.linalg.norm(center - prev_center) / dt)
            vel_fis.append(fi)
        prev_center = center
    avg_v_ms = np.mean(velocities) / 1000.0 if velocities else 0

    # 为可视化预计算每帧的速度向量（2D 像素空间，多帧平滑）
    SCALE_PX_TO_M = (BALL_RADIUS_MM * 2) / (median_r * 2) / 1000.0
    frame_vel_2d = {}  # {fi: (vx_mps, vy_mps)}
    SMOOTH_WIN = 3
    for idx, fi in enumerate(labeled_frames):
        half = SMOOTH_WIN // 2
        w0 = max(0, idx - half)
        w1 = min(len(labeled_frames), idx + half + 1)
        if w1 - w0 >= 2:
            f0, f1 = labeled_frames[w0], labeled_frames[w1-1]
            cx0, cy0, _ = detections[f0]
            cx1, cy1, _ = detections[f1]
            t_span = (f1 - f0) * dt
            if t_span > 0:
                vx = ((cx1 - cx0) / t_span) * SCALE_PX_TO_M
                vy = ((cy1 - cy0) / t_span) * SCALE_PX_TO_M
                frame_vel_2d[fi] = (vx, vy)

    # ── 控制台报告 ──
    total_dot_ids = set()
    for fi in all_annotations:
        total_dot_ids.update(all_annotations[fi].keys())

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Labeled frames:       {len(all_annotations)}")
    print(f"  Unique dot IDs:       {sorted(total_dot_ids)}")
    print(f"  Reliable pairs (3+):  {n_reliable}")
    print(f"  Fallback pairs (2):   {n_fallback}")
    print(f"  Total rotation:       {np.degrees(total_theta):.1f} deg")
    print(f"  Rotation RPM:         {rpm:.1f}")
    print(f"  Rotation RPS:         {rps:.2f}")
    print(f"  Axis (user coords):   [{axis_user[0]:.4f}, {axis_user[1]:.4f}, {axis_user[2]:.4f}]")
    print(f"  Linear velocity:      {avg_v_ms:.2f} m/s ({avg_v_ms*3.6:.1f} km/h)")
    if reliability_note:
        print(f"  {reliability_note}")
    print(f"{'='*55}")

    if pair_details:
        print(f"\n  Per-pair details (signed degrees, after axis correction):")
        for pfi, cfi, sdeg, nids, rel in pair_details:
            tag = "✓" if rel else "⚠"
            print(f"    F{pfi} -> F{cfi}: {sdeg:+.2f} deg ({nids} dots) {tag}")

    # 旋转类型
    ax_x, ax_y, ax_z = axis_user
    if abs(ax_x) > abs(ax_y) and abs(ax_x) > abs(ax_z):
        spin_type = "Topspin" if ax_x > 0 else "Backspin"
    elif abs(ax_y) > abs(ax_x) and abs(ax_y) > abs(ax_z):
        spin_type = "Sidespin-R" if ax_y > 0 else "Sidespin-L"
    else:
        spin_type = "Gyro/Spiral"
    print(f"  Spin type:            {spin_type}")

    # ═════════════════════════════════════════════════════
    #  可视化回放
    # ═════════════════════════════════════════════════════

    print(f"\n[Viewer] Drag trackbar | Q/ESC to quit")
    print(f"  Blue circle = ball | Colored dots = labeled points")
    print(f"  Red arrow = velocity | Yellow line = rotation axis")

    win2 = "Spin Results — Labeled Dots + Velocity + Axis"
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    disp_w2 = min(1600, video_w)
    disp_h2 = int(disp_w2 * video_h / video_w)
    cv2.resizeWindow(win2, disp_w2, disp_h2 + 40)
    cv2.createTrackbar("Frame", win2, labeled_frames[0], labeled_frames[-1], lambda x: None)
    cv2.createTrackbar("Thresh", win2, 0, 255, lambda x: None)

    ax_img_x = axis_user[1]   # user y → image x
    ax_img_y = axis_user[0]   # user x → image y

    while True:
        fi = cv2.getTrackbarPos("Frame", win2)
        # 吸附到最近的有标注帧
        if fi not in all_annotations:
            nearest = min(labeled_frames, key=lambda x: abs(x-fi))
            cv2.setTrackbarPos("Frame", win2, nearest)
            fi = nearest

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        display = frame.copy()

        if fi in detections:
            cx, cy, r = detections[fi]

            # 球框
            cv2.circle(display, (int(cx), int(cy)), int(r), (255, 80, 0), 2)

            # 标注点
            dots = all_annotations.get(fi, {})
            for dot_id, (px, py) in dots.items():
                color = DOT_COLORS[dot_id % len(DOT_COLORS)]
                cv2.circle(display, (int(px), int(py)), 4, color, -1)
                cv2.circle(display, (int(px), int(py)), 6, color, 1)
                cv2.putText(display, str(dot_id), (int(px)+7, int(py)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 速度箭头（红色）
            if fi in frame_vel_2d:
                vx, vy = frame_vel_2d[fi]
                vmag = np.hypot(vx, vy)
                if vmag > 0.1:
                    arrow_scale = 60.0  # 1 m/s ≈ 60 px
                    alen = min(vmag * arrow_scale, r * 3.0)
                    vxn, vyn = vx/vmag*alen, vy/vmag*alen
                    tipx, tipy = int(cx+vxn), int(cy+vyn)
                    cv2.arrowedLine(display, (int(cx), int(cy)), (tipx, tipy),
                                    (0, 0, 255), 2, tipLength=0.25)
                    cv2.putText(display, f'{vmag:.1f} m/s', (tipx+5, tipy-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 旋转轴（黄色虚线，贯穿球心）
            if np.hypot(ax_img_x, ax_img_y) > 0.01:
                ax_len = r * 1.5
                ax_n = np.hypot(ax_img_x, ax_img_y)
                dx = ax_img_x / ax_n * ax_len
                dy = ax_img_y / ax_n * ax_len
                cv2.line(display, (int(cx-dx), int(cy-dy)), (int(cx+dx), int(cy+dy)),
                         (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(display, (int(cx+dx), int(cy+dy)), 4, (0, 255, 255), -1)

            # 在主视图上画一个1像素中心标记
            cv2.circle(display, (int(cx), int(cy)), 1, (0, 255, 255), -1)

            # 生成并贴上二值化面板
            try:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thr2 = cv2.getTrackbarPos("Thresh", win2)
                bin_img2 = binarize_ball_region(gray_full, cx, cy, r, out_size=150, thresh=thr2)
                bh2, bw2 = bin_img2.shape[:2]
                mx = 10
                my = 10
                x1 = max(0, display.shape[1] - bw2 - mx)
                y1 = my
                display[y1:y1+bh2, x1:x1+bw2] = bin_img2
                cv2.rectangle(display, (x1-1, y1-1), (x1+bw2+1, y1+bh2+1), (200,200,200), 1)
            except Exception:
                pass

        # 面板
        n_my_dots = len(all_annotations.get(fi, {}))
        lines = [
            f"Frame: {fi} | Dots: {n_my_dots} | {spin_type}",
            f"RPM: {rpm:.0f} | RPS: {rps:.1f} | Vel: {avg_v_ms:.2f} m/s",
            f"Axis: [{axis_user[0]:.3f}, {axis_user[1]:.3f}, {axis_user[2]:.3f}]",
            f"Red arrow = velocity | Yellow line = rotation axis",
        ]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (520, len(lines)*22+15), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)
        for i, line in enumerate(lines):
            cv2.putText(display, line, (10, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win2, display)
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
