"""旋转计算: 3帧滑动窗口 Kabsch + 中值滤波"""
import numpy as np
from .geometry import kabsch


def compute_rotation(frames_3d, frame_labels, valid_fi, fps):
    """3帧窗口 Kabsch + 中值滤波 → RPM, 旋转轴"""
    dt = 1.0 / fps
    all_thetas = []
    all_axes = []
    global_ref_axis = None

    for i in range(len(valid_fi) - 2):
        f0, f1, f2 = valid_fi[i], valid_fi[i + 1], valid_fi[i + 2]
        ids0 = set(frame_labels[f0].values())
        ids1 = set(frame_labels[f1].values())
        ids2 = set(frame_labels[f2].values())
        triple = ids0 & ids1 & ids2
        if len(triple) < 3:
            continue

        gids = sorted(triple)
        inv0 = {gid: lidx for lidx, gid in frame_labels[f0].items()}
        inv1 = {gid: lidx for lidx, gid in frame_labels[f1].items()}
        inv2 = {gid: lidx for lidx, gid in frame_labels[f2].items()}
        pts0 = np.array([frames_3d[f0][inv0[g]] for g in gids])
        pts1 = np.array([frames_3d[f1][inv1[g]] for g in gids])
        pts2 = np.array([frames_3d[f2][inv2[g]] for g in gids])

        try:
            _, theta_01, axis_01 = kabsch(pts0, pts1)
            _, theta_12, axis_12 = kabsch(pts1, pts2)

            if global_ref_axis is None:
                global_ref_axis = axis_01
            if np.dot(global_ref_axis, axis_01) < 0:
                axis_01 = -axis_01; theta_01 = -theta_01
            if np.dot(global_ref_axis, axis_12) < 0:
                axis_12 = -axis_12; theta_12 = -theta_12

            all_thetas.append(theta_01)
            all_thetas.append(theta_12)
            all_axes.append(axis_01)
            all_axes.append(axis_12)
        except Exception:
            pass

    if len(all_thetas) < 3:
        return 0.0, 0.0, np.array([0., 0., 1.]), "N/A", "N/A"

    thetas = np.abs(all_thetas)
    med = np.median(thetas)
    mad = np.median(np.abs(thetas - med))
    keep = np.abs(thetas - med) < 2.0 * (mad + 1e-9)
    filtered_thetas = thetas[keep]
    filtered_axes = [all_axes[j] for j in range(len(thetas)) if keep[j]]

    if len(filtered_thetas) < 2:
        return 0.0, 0.0, np.array([0., 0., 1.]), "N/A", "N/A"

    mean_theta = np.mean(filtered_thetas)
    omega = mean_theta / dt
    rps = omega / (2 * np.pi)
    rpm = rps * 60

    avg_axis = np.mean(filtered_axes, axis=0)
    avg_axis /= (np.linalg.norm(avg_axis) + 1e-9)

    ax_x, ax_y, ax_z = avg_axis
    if abs(ax_x) > abs(ax_y) and abs(ax_x) > abs(ax_z):
        spin_type = "Topspin" if ax_x > 0 else "Backspin"
    elif abs(ax_y) > abs(ax_x) and abs(ax_y) > abs(ax_z):
        spin_type = "Sidespin-R" if ax_y > 0 else "Sidespin-L"
    else:
        spin_type = "Gyro/Spiral"
    cw = "CCW" if avg_axis[2] > 0 else "CW"

    return rpm, rps, avg_axis, cw, spin_type
