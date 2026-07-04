"""可视化: 交互式帧浏览 + 3D球面旋转轴"""
import cv2, numpy as np
from .config import DOT_COLORS
from .geometry import kabsch


def show_3d_axis(frames_3d, frame_labels, axis_user, cw, valid_fi):
    """弹出 3D 球面 + 旋转轴"""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    mid_fi = valid_fi[len(valid_fi) // 2]
    labels = frame_labels[mid_fi]
    dots_3d = frames_3d[mid_fi]

    xs, ys, zs = [], [], []
    for v in dots_3d:
        xs.append(v[0]); ys.append(v[1]); zs.append(v[2])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 30)
    ax.plot_wireframe(
        np.outer(np.sin(phi), np.cos(theta)),
        np.outer(np.sin(phi), np.sin(theta)),
        np.outer(np.cos(phi), np.ones_like(theta)),
        color='gray', alpha=0.15, linewidth=0.3)

    ax.scatter(xs, ys, zs, c='red', s=50, depthshade=True)
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        gid = labels.get(i, i)
        ax.text(x, y, z, str(gid), color='white', fontsize=8,
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.6))

    ax.quiver(0, 0, 0, axis_user[0], axis_user[1], axis_user[2],
              color='cyan', linewidth=3, arrow_length_ratio=0.15, label=f'Axis ({cw})')
    ax.quiver(0, 0, 0, -axis_user[0], -axis_user[1], -axis_user[2],
              color='cyan', linewidth=1, alpha=0.3, arrow_length_ratio=0.15)

    ax.set_title(f"Rotation Axis: [{axis_user[0]:.3f}, {axis_user[1]:.3f}, {axis_user[2]:.3f}] ({cw})")
    ax.set_xlabel('X (right)'); ax.set_ylabel('Y (down)'); ax.set_zlabel('Z (forward)')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=90, azim=0)
    ax.legend()
    plt.show()


def interactive_viewer(images, use_indices, frames_2d, frames_3d, frames_ball,
                       frame_labels, axis_user, rpm, rps, spin_type, cw, valid_fi,
                       matched_gids=None, fps=None):
    """
    交互式逐帧浏览窗口。matched_gids={fi: set(gid)} — 成功匹配的点。
    空心同色圆圈 = 上一帧点根据旋转矩阵预测到当前帧的位置。
    灰色点 = 未匹配的点 (新 ID, 没有前帧对应)。
    """
    win = "spinCal — Result Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 800)

    # 角速度 (rad/s) — 预测时按实际帧间隔换算
    omega = None
    if rpm > 0 and fps is not None:
        omega = rpm / 60.0 * 2.0 * np.pi

    ax_ix, ax_iy = axis_user[0], axis_user[1]
    idx = 0

    while idx < len(valid_fi):
        fi = valid_fi[idx]
        img_idx = use_indices.index(fi) if fi in use_indices else 0
        img = images[img_idx]
        if img is None: idx += 1; continue
        h, w = img.shape[:2]
        display = cv2.resize(img, (w * 6, h * 6), interpolation=cv2.INTER_NEAREST)

        bcx, bcy, br, _ = frames_ball[fi]
        cv2.circle(display, (int(bcx * 6), int(bcy * 6)), int(br * 6), (255, 100, 0), 1)

        dots_2d = frames_2d[fi]
        labels = frame_labels[fi]
        matched = matched_gids.get(fi, set()) if matched_gids else set()
        for k, (dx, dy) in enumerate(dots_2d):
            gid = labels.get(k, k)
            is_matched = gid in matched
            c = DOT_COLORS[gid % len(DOT_COLORS)]
            if is_matched:
                cv2.circle(display, (int(dx * 6), int(dy * 6)), 3, c, -1)
                cv2.putText(display, str(gid), (int(dx * 6) + 5, int(dy * 6) - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)
            else:
                cv2.circle(display, (int(dx * 6), int(dy * 6)), 2, (140, 140, 140), -1)

        # 空心预测点: 上一帧所有点经旋转 R 投影 (R 按实际帧间隔换算)
        n_pred = 0; n_back = 0; n_prev = 0
        if idx > 0:
            prev_fi = valid_fi[idx - 1]
            prev_3d_all = frames_3d.get(prev_fi, [])
            n_prev = len(prev_3d_all)
            if omega is not None and n_prev > 0:
                pair_gap = fi - prev_fi
                R, _ = cv2.Rodrigues(axis_user * omega * pair_gap / fps)
                cur_matched = matched_gids.get(fi, set()) if matched_gids else set()
                prev_labels = frame_labels.get(prev_fi, {})
                for k, v3d in enumerate(prev_3d_all):
                    pred_v = R @ v3d
                    pred_v = pred_v / (np.linalg.norm(pred_v) + 1e-9)  # 强制归一化
                    if pred_v[2] > 0:
                        n_back += 1; continue
                    pred_x = bcx + pred_v[0] * br
                    pred_y = bcy + pred_v[1] * br
                    gid_prev = prev_labels.get(k, -1)
                    in_use = gid_prev in cur_matched
                    if in_use:
                        c_pred = DOT_COLORS[abs(gid_prev) % len(DOT_COLORS)]
                        cv2.circle(display, (int(pred_x * 6), int(pred_y * 6)), 5, c_pred, 2)
                        cv2.putText(display, str(gid_prev),
                                    (int(pred_x * 6) + 6, int(pred_y * 6) - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, c_pred, 1)
                    else:
                        cv2.circle(display, (int(pred_x * 6), int(pred_y * 6)), 4, (120,120,120), 1)
                    n_pred += 1

        if np.hypot(ax_ix, ax_iy) > 0.01:
            al = br * 6 * 1.5
            an = np.hypot(ax_ix, ax_iy)
            dxa, dya = ax_ix / an * al, ax_iy / an * al
            cx6, cy6 = int(bcx * 6), int(bcy * 6)
            cv2.line(display, (cx6 - int(dxa), cy6 - int(dya)),
                     (cx6 + int(dxa), cy6 + int(dya)), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(display, (cx6 + int(dxa), cy6 + int(dya)), 4, (0, 255, 255), -1)

        # ── 逐帧对旋转测量 ──
        pair_rpm = 0.0; pair_axis = np.array([0.,0.,0.]); pair_gap = 0
        if idx > 0 and fps is not None:
            prev_fi = valid_fi[idx - 1]
            pair_gap = fi - prev_fi
            labels_prev = frame_labels.get(prev_fi, {})
            labels_cur = frame_labels.get(fi, {})
            ids_prev = set(labels_prev.values())
            ids_cur = set(labels_cur.values())
            common = ids_prev & ids_cur
            if len(common) >= 2:
                gids = sorted(common)
                inv_p = {gid: lidx for lidx, gid in labels_prev.items()}
                inv_c = {gid: lidx for lidx, gid in labels_cur.items()}
                pts_p = np.array([frames_3d[prev_fi][inv_p[g]] for g in gids])
                pts_c = np.array([frames_3d[fi][inv_c[g]] for g in gids])
                try:
                    _, theta_p, axis_p = kabsch(pts_p, pts_c)
                    actual_dt = pair_gap / fps
                    pair_rpm = abs(theta_p) / (2 * np.pi) * 60 / actual_dt
                    pair_axis = axis_p
                except Exception:
                    pass

        if idx == 0:
            pred_info = "prev=0"
        elif omega is None:
            pred_info = f"NO_ROT(rpm={rpm:.0f},fps={fps})"
        else:
            pred_info = f"prev={n_prev} back={n_back} pred={n_pred}"
        lines = [
            f"F{fi} | Dots:{len(dots_2d)} | {pred_info} | {spin_type} | {cw}",
            f"RPM: {rpm:.0f} | RPS: {rps:.1f}",
            f"Axis: [{axis_user[0]:.3f},{axis_user[1]:.3f},{axis_user[2]:.3f}]",
        ]
        if pair_rpm > 0:
            lines.append(f"F{prev_fi}→F{fi} (gap={pair_gap}): {pair_rpm:.0f}RPM "
                         f"axis=[{pair_axis[0]:.2f},{pair_axis[1]:.2f},{pair_axis[2]:.2f}] "
                         f"({len(common)}dots)")
        ov = display.copy()
        cv2.rectangle(ov, (0, 0), (400, len(lines) * 18 + 10), (0, 0, 0), -1)
        display = cv2.addWeighted(ov, 0.5, display, 0.5, 0)
        for i, line in enumerate(lines):
            cv2.putText(display, line, (3, 14 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(0) & 0xFF
        if key == 13: idx += 1
        elif key == 27 or key == ord('q'): break
        elif key == ord('b'): idx = max(0, idx - 1)
        elif key == ord('m') or key == ord('M'):
            show_3d_axis(frames_3d, frame_labels, axis_user, cw, valid_fi)

    cv2.destroyAllWindows()
