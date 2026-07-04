"""旋转估计: 逐帧对 RANSAC Kabsch + 共识聚合 + ICP/轨迹精炼。

架构 (单次流水线, 无外环 — 遵循刚体物理约束, 保留转速衰减/转轴进动):
  Stage 1  3D 邻近匹配 (宽阈值)        → labels
  Stage 2  粗旋转估计 (estimate_spin)  → 初始 rpm/axis
  Stage 3  ICP 旋转约束重匹配 (紧阈值) → 精炼 labels (失败回退 Stage 2)
  Stage 4  轨迹一致性剔点              → 清洗 labels
  Stage 5  最终 estimate_spin + 误差   → 最终 rpm/axis/err

核心: 逐帧对 (fi, fi+1) 独立 RANSAC Kabsch (≥3 公共点), 一次一份 inlier mask,
      无 3 帧窗口的键碰撞/重复计算。聚合用 RANSAC 共识 + 加权平均。
"""
import cv2, numpy as np
from .config import (MATCH_MAX_DISP, ICP_MATCH_DISP, TRAJ_RMS_THRESH,
                     MIN_COMMON_DOTS)
from .geometry import kabsch
from .matching import track_dots_across_frames

RANSAC_ITER = 50
RANSAC_INLIER_TOL = 0.05   # 单位球面上弦长 (~2.9° 弧)
CONSENSUS_REL_TOL = 0.15   # omega 共识相对容差


# ═══════════════════ RANSAC Kabsch ═══════════════════

def ransac_kabsch(ps, pd, inlier_tol=RANSAC_INLIER_TOL, max_iter=RANSAC_ITER):
    """
    随机抽样 3 对点 → Kabsch → 统计内点 → 保留最优 → 内点重拟合。
    返回 (R, theta, axis, inlier_mask)。
    """
    n = len(ps)
    if n < MIN_COMMON_DOTS:
        return None, 0.0, np.array([0., 0., 1.]), np.zeros(n, dtype=bool)

    best_inliers = 0
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(max_iter):
        idx = np.random.choice(n, size=min(3, n), replace=False)
        try:
            R_guess, _, _ = kabsch(ps[idx], pd[idx])
        except Exception:
            continue
        pred = (R_guess @ ps.T).T
        dists = np.linalg.norm(pred - pd, axis=1)
        mask = dists < inlier_tol
        n_in = np.sum(mask)
        if n_in > best_inliers:
            best_inliers = n_in
            best_mask = mask

    if best_inliers < MIN_COMMON_DOTS:
        return None, 0.0, np.array([0., 0., 1.]), best_mask

    R, theta, axis = kabsch(ps[best_mask], pd[best_mask])
    return R, theta, axis, best_mask


# ═══════════════════ RANSAC 1D 共识 ═══════════════════

def ransac_consensus_1d(values, weights, relative_tol=CONSENSUS_REL_TOL,
                         max_iter=100):
    """
    在 1D 值中找最大加权共识簇 (替代 MAD 中值滤波, 不假设对称分布)。
    返回 inlier mask (bool 数组)。
    """
    n = len(values)
    if n < 3:
        return np.ones(n, dtype=bool)

    best_score = 0.0
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        idx = np.random.randint(0, n)
        center = values[idx]
        tol = relative_tol * max(abs(center), 1e-9)
        inliers = np.abs(values - center) < tol
        score = np.sum(weights[inliers]) if weights is not None else np.sum(inliers)
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if np.sum(best_inliers) < 2:
        return np.ones(n, dtype=bool)
    return best_inliers


# ═══════════════════ 逐帧对旋转估计 ═══════════════════

def estimate_per_pair_rotations(frames_3d, frame_labels, valid_fi, fps):
    """
    对每个相邻帧对 (fi, fi+1) 做 RANSAC Kabsch (≥MIN_COMMON_DOTS 公共点)。
    返回 pairs 列表, 每项含 {f0,f1,gap,omega,axis,n_inliers,n_common,rps}。
    omega 为有符号 rad/s (axis 已对齐 global_ref_axis)。
    """
    pairs = []
    global_ref_axis = None

    for i in range(len(valid_fi) - 1):
        f0, f1 = valid_fi[i], valid_fi[i + 1]
        gap = f1 - f0
        ids0 = set(frame_labels[f0].values())
        ids1 = set(frame_labels[f1].values())
        common = ids0 & ids1
        if len(common) < MIN_COMMON_DOTS:
            continue

        gids = sorted(common)
        inv0 = {gid: lidx for lidx, gid in frame_labels[f0].items()}
        inv1 = {gid: lidx for lidx, gid in frame_labels[f1].items()}
        pts0 = np.array([frames_3d[f0][inv0[g]] for g in gids])
        pts1 = np.array([frames_3d[f1][inv1[g]] for g in gids])

        R, theta, axis, mask = ransac_kabsch(pts0, pts1)
        if R is None:
            continue

        # 符号一致性: 对齐到首个轴
        if global_ref_axis is None:
            global_ref_axis = axis
        if np.dot(global_ref_axis, axis) < 0:
            axis = -axis
            theta = -theta

        actual_dt = gap / fps
        omega = theta / actual_dt  # rad/s, 有符号
        n_in = int(np.sum(mask))
        pairs.append({
            'f0': f0, 'f1': f1, 'gap': gap,
            'omega': omega, 'axis': axis,
            'n_inliers': n_in, 'n_common': len(gids),
            'rps': abs(omega) / (2 * np.pi),
        })
    return pairs


def aggregate_rotation(pairs):
    """
    RPM = 中位 |omega| (小 N 下比 RANSAC 聚类更稳); 拒绝 >2x/<0.5x 中位的离群。
    轴 = 内点加权平均 (权重=n_inliers)。
    返回 (rpm, rps, axis, cw, spin_type)。
    """
    if len(pairs) < 2:
        return 0.0, 0.0, np.array([0., 0., 1.]), "N/A", "N/A"

    abs_omegas = np.abs([p['omega'] for p in pairs])
    med_omega = np.median(abs_omegas)
    # 拒绝明显离群 (方向保留: 用原始 pairs 的 keep 掩码)
    keep = (abs_omegas > 0.5 * med_omega) & (abs_omegas < 2.0 * med_omega)
    if np.sum(keep) < 2:
        keep = np.ones(len(pairs), dtype=bool)

    # RPM: 中位 |omega| (鲁棒, 不被高 RPM 簇拉偏)
    omega = np.median(abs_omegas[keep])
    rps = omega / (2 * np.pi)
    rpm = rps * 60

    # 轴: 内点加权平均
    filt_axes = np.array([pairs[j]['axis'] for j in range(len(pairs)) if keep[j]])
    filt_weights = np.array([pairs[j]['n_inliers'] for j in range(len(pairs)) if keep[j]],
                            dtype=np.float64)
    if np.sum(filt_weights) < 1e-9:
        return rpm, rps, np.array([0., 0., 1.]), "N/A", "N/A"
    axis = np.average(filt_axes, weights=filt_weights, axis=0)
    axis /= (np.linalg.norm(axis) + 1e-9)

    ax_x, ax_y, ax_z = axis
    if abs(ax_x) > abs(ax_y) and abs(ax_x) > abs(ax_z):
        spin_type = "Topspin" if ax_x > 0 else "Backspin"
    elif abs(ax_y) > abs(ax_x) and abs(ax_y) > abs(ax_z):
        spin_type = "Sidespin-R" if ax_y > 0 else "Sidespin-L"
    else:
        spin_type = "Gyro/Spiral"
    cw = "CCW" if ax_z > 0 else "CW"

    return rpm, rps, axis, cw, spin_type


def estimate_spin(frames_3d, frame_labels, valid_fi, fps):
    """逐帧对 RANSAC + 聚合。返回 (rpm, rps, axis, cw, spin_type, pairs)。"""
    pairs = estimate_per_pair_rotations(frames_3d, frame_labels, valid_fi, fps)
    rpm, rps, axis, cw, spin_type = aggregate_rotation(pairs)
    return rpm, rps, axis, cw, spin_type, pairs


# ═══════════════════ 点轨迹一致性检验 ═══════════════════

def remove_inconsistent_dots(frames_3d, frame_labels, valid_fi,
                               rpm, axis, fps, max_rms=TRAJ_RMS_THRESH):
    """
    对每个全局点 ID, 检验其跨帧轨迹是否与全局旋转一致。
    仅剔除"相对离群"的轨迹 (RMS > max(max_rms, 2×中位RMS)) —
    这样当全局轴估计有偏差时(所有点RMS都偏高), 不会误删好点。
    返回 (cleaned_labels, n_removed)。
    """
    dot_traj = {}
    for fi in valid_fi:
        for lidx, gid in frame_labels.get(fi, {}).items():
            dot_traj.setdefault(gid, []).append((fi, lidx, frames_3d[fi][lidx]))

    omega = rpm / 60.0 * 2.0 * np.pi
    rot_per_frame = axis * omega / fps   # rad/帧索引步

    rms_by_gid = {}
    for gid, traj in dot_traj.items():
        if len(traj) < 3:
            continue
        traj_sorted = sorted(traj, key=lambda x: x[0])
        errors = []
        for k in range(len(traj_sorted) - 1):
            fi_a, _, pos_a = traj_sorted[k]
            fi_b, _, pos_b = traj_sorted[k + 1]
            gap = fi_b - fi_a
            R, _ = cv2.Rodrigues(rot_per_frame * gap)
            pred = R @ pos_a
            pred = pred / (np.linalg.norm(pred) + 1e-9)
            errors.append(np.linalg.norm(pred - pos_b))
        rms_by_gid[gid] = np.sqrt(np.mean(np.array(errors) ** 2))

    if not rms_by_gid:
        return frame_labels, 0

    # 相对阈值: 仅当某点 RMS 远高于群体中位时才剔 (容忍全局轴偏差)
    med_rms = np.median(list(rms_by_gid.values()))
    thresh = max(max_rms, 2.0 * med_rms)
    bad_gids = {gid for gid, r in rms_by_gid.items() if r > thresh}

    if not bad_gids:
        return frame_labels, 0

    cleaned = {fi: {lidx: gid for lidx, gid in frame_labels[fi].items()
                    if gid not in bad_gids} for fi in valid_fi}
    return cleaned, len(bad_gids)


# ═══════════════════ matched_gids 重建 ═══════════════════

def rebuild_matched_gids(frame_labels, valid_fi):
    """
    O(n) 重建 matched_gids: {fi: set(与上一帧共有的 gid)}。
    """
    mg = {}
    for pos, fi in enumerate(valid_fi):
        if pos == 0:
            mg[fi] = set()
        else:
            prev_fi = valid_fi[pos - 1]
            mg[fi] = set(frame_labels[fi].values()) & set(frame_labels[prev_fi].values())
    return mg


# ═══════════════════ 测量误差估计 ═══════════════════

def measurement_error(pairs, fps):
    """
    误差估计 (鲁棒, 与 aggregate_rotation 一致地用 2× 中位过滤):
      - RPS: 中位 ± 95% CI (中位标准误), robust-std (MAD), CV — 描述转速置信度
      - 轴 RMS: 各共识帧对轴与共识轴的夹角 RMS (度) — 描述轴向置信度

    相比旧版 (极差 / max pairwise 夹角): 不被单个离群对拉偏; CI 反映中位估计
    本身的不确定度 (随样本数 N 收敛), 而非单次测量的散布。
    """
    if len(pairs) < 2:
        return None
    n = len(pairs)
    rps_arr = np.array([p['rps'] for p in pairs])
    axes = np.array([p['axis'] for p in pairs])

    # 与 aggregate 一致的 2× 中位过滤 (描述估计所依据的内点集)
    med = np.median(rps_arr)
    keep = (rps_arr > 0.5 * med) & (rps_arr < 2.0 * med)
    if np.sum(keep) < 2:
        keep = np.ones(n, dtype=bool)
    n_keep = int(np.sum(keep))
    rps_k = rps_arr[keep]
    axes_k = axes[keep]

    # 共识轴 = 内点旋转向量 (axis*omega/fps) 均值的方向
    rotvecs_k = np.array([pairs[j]['axis'] * pairs[j]['omega'] / fps
                          for j in range(n) if keep[j]])
    cons_rv = np.mean(rotvecs_k, axis=0)
    cons_axis = cons_rv / (np.linalg.norm(cons_rv) + 1e-9)

    # RPS 鲁棒统计 (内点集)
    med_rps = np.median(rps_k)
    mad_rps = np.median(np.abs(rps_k - med_rps))
    robust_std = 1.4826 * mad_rps
    cv = robust_std / (med_rps + 1e-9)
    se_median = 1.253 * robust_std / np.sqrt(n_keep)   # 中位的标准误
    ci95 = 1.96 * se_median

    # 轴 RMS 偏差 (abs 处理符号二义)
    axis_angles = np.degrees(np.arccos(np.clip(np.abs(axes_k @ cons_axis), 0.0, 1.0)))
    axis_rms = np.sqrt(np.mean(axis_angles ** 2))

    return {
        'n': n, 'n_keep': n_keep,
        'frame_gaps': [p['gap'] for p in pairs],
        'rps_median': med_rps,
        'rps_robust_std': robust_std,
        'cv': cv,
        'rps_ci95': ci95,
        'axis_rms_deg': axis_rms,
    }


# ═══════════════════ 主流水线 ═══════════════════

def run_pipeline(frames_3d, valid_fi, fps):
    """
    5 阶段单次流水线 (无外环)。
    返回 (rpm, rps, axis, cw, spin_type, frame_labels, matched_gids, err)。
    """
    fi_deltas = [valid_fi[i + 1] - valid_fi[i] for i in range(len(valid_fi) - 1)]
    avg_gap = np.mean(fi_deltas) if fi_deltas else 1.0
    print(f"  [DIAG] {len(valid_fi)} frames, avg gap={avg_gap:.1f}, "
          f"max={max(fi_deltas) if fi_deltas else 0}")

    # ── Stage 1: 3D 邻近匹配 ──
    print("  [Stage 1] proximity matching...")
    labels, _, matched_gids = track_dots_across_frames(
        frames_3d, valid_fi, max_displacement=MATCH_MAX_DISP)

    # ── Stage 2: 粗旋转 ──
    rpm, rps, axis, cw, st, pairs = estimate_spin(frames_3d, labels, valid_fi, fps)
    print(f"  [Stage 2] rough: RPM={rpm:.0f}, {len(pairs)} pairs")
    if rpm == 0.0:
        err = measurement_error(pairs, fps)
        _print_error(err)
        return rpm, rps, axis, cw, st, labels, matched_gids, err

    # ── Stage 3: ICP 旋转约束重匹配 (紧阈值) ──
    print("  [Stage 3] ICP re-matching (tight threshold)...")
    omega = rpm / 60.0 * 2.0 * np.pi
    rot_per_frame = axis * omega * avg_gap / fps
    labels_icp, _, matched_gids_icp = track_dots_across_frames(
        frames_3d, valid_fi, max_displacement=ICP_MATCH_DISP,
        rot_per_frame=rot_per_frame)
    rpm_i, rps_i, axis_i, cw_i, st_i, pairs_i = estimate_spin(
        frames_3d, labels_icp, valid_fi, fps)
    print(f"  [Stage 3] ICP: RPM={rpm_i:.0f}, {len(pairs_i)} pairs")

    # 失败回退: ICP 清零则保留 Stage 2
    if rpm_i == 0:
        print("  [Stage 3] ICP failed (rpm=0), keeping Stage 2 result")
        labels_icp, matched_gids_icp = labels, matched_gids
        rpm_i, rps_i, axis_i, cw_i, st_i = rpm, rps, axis, cw, st

    # ── Stage 4: 轨迹一致性剔点 ──
    labels_cur, n_removed = remove_inconsistent_dots(
        frames_3d, labels_icp, valid_fi, rpm_i, axis_i, fps)
    if n_removed > 0:
        print(f"  [Stage 4] removed {n_removed} inconsistent trajectories")
        labels_final = labels_cur
        matched_gids_final = rebuild_matched_gids(labels_final, valid_fi)
    else:
        labels_final = labels_icp
        matched_gids_final = matched_gids_icp

    # ── Stage 5: 最终旋转 + 误差 ──
    rpm_f, rps_f, axis_f, cw_f, st_f, pairs_f = estimate_spin(
        frames_3d, labels_final, valid_fi, fps)
    # 失败回退: 最终清零 或 剔点后样本过少 → 保留 Stage 3 结果
    if rpm_f == 0 or len(pairs_f) < 3:
        print(f"  [Stage 5] final unreliable (rpm={rpm_f:.0f}, {len(pairs_f)} pairs), "
              f"keeping Stage 3 result")
        rpm_f, rps_f, axis_f, cw_f, st_f = rpm_i, rps_i, axis_i, cw_i, st_i
        pairs_f = pairs_i
        labels_final, matched_gids_final = labels_icp, matched_gids_icp
    else:
        print(f"  [Stage 5] final: RPM={rpm_f:.0f}, {len(pairs_f)} pairs")

    err = measurement_error(pairs_f, fps)
    _print_error(err)
    print(f"  [Final] RPM={rpm_f:.0f}, RPS={rps_f:.2f}, {cw_f}, {st_f}")
    return rpm_f, rps_f, axis_f, cw_f, st_f, labels_final, matched_gids_final, err


def _print_error(err):
    if err is None:
        print("  [ERROR EST] insufficient pairs")
        return
    gaps = err['frame_gaps']
    unique_gaps = sorted(set(gaps)) if gaps else []
    gap_info = (" (all consecutive)" if len(unique_gaps) == 1 and unique_gaps == [1]
                else f" (gaps: {unique_gaps})") if unique_gaps else ""
    print(f"  [ERROR EST] {err['n']} pairs{gap_info}, {err['n_keep']} in consensus")
    print(f"  [ERROR EST] RPS: {err['rps_median']:.2f} ± {err['rps_ci95']:.2f} (95% CI of median)  "
          f"per-pair σ~{err['rps_robust_std']:.1f} ({100 * err['cv']:.0f}% CV)")
    print(f"  [ERROR EST] Axis: {err['axis_rms_deg']:.1f}° RMS deviation from consensus")
