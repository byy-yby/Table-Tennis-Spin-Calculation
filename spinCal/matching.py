"""3D 邻近匹配 + ICP 旋转修正匹配 + 跨帧 ID 追踪"""
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from .config import MATCH_MAX_DISP
from .geometry import kabsch


def match_by_3d_proximity(prev_3d, cur_3d, max_displacement=None):
    """基于 3D 空间绝对距离的匈牙利匹配"""
    if max_displacement is None: max_displacement = MATCH_MAX_DISP
    n_prev, n_cur = len(prev_3d), len(cur_3d)
    if n_prev == 0 or n_cur == 0:
        return []

    cost = np.full((n_prev, n_cur), 1e9)
    for i in range(n_prev):
        for j in range(n_cur):
            dist = np.linalg.norm(prev_3d[i] - cur_3d[j])
            if dist < max_displacement:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(row_ind, col_ind) if cost[i, j] < max_displacement]


def _match_with_R(R, prev_3d, cur_3d, max_displacement):
    """
    用旋转 R 预测 prev 位置 → 匈牙利匹配 cur。
    背面点 (预测 z>0) 不参与。返回匹配对列表 [(prev_idx, cur_idx)]。
    """
    n_prev, n_cur = len(prev_3d), len(cur_3d)
    if n_prev == 0 or n_cur == 0:
        return []

    pred_3d = np.array([R @ p for p in prev_3d])
    cost = np.full((n_prev, n_cur), 1e9)
    for i in range(n_prev):
        if pred_3d[i][2] > 0:          # 背面不可见
            continue
        for j in range(n_cur):
            dist = np.linalg.norm(pred_3d[i] - cur_3d[j])
            if dist < max_displacement:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(row_ind, col_ind) if cost[i, j] < max_displacement]


def match_by_rotation_icp(prev_3d, cur_3d, initial_R,
                           max_displacement=None, max_iter=5):
    """
    ICP 迭代修正匹配:
    1. 用当前 R 预测位置 → 匈牙利匹配
    2. 从匹配对提取对应点 → Kabsch 修正 R
    3. 重复直到 R 收敛 (变化 < 1e-4) 或达到最大迭代
    """
    if max_displacement is None: max_displacement = MATCH_MAX_DISP
    n_prev, n_cur = len(prev_3d), len(cur_3d)
    if n_prev == 0 or n_cur == 0:
        return []

    current_R = initial_R.copy()

    for _ in range(max_iter):
        pairs = _match_with_R(current_R, prev_3d, cur_3d, max_displacement)
        if len(pairs) < 2:
            break

        src = np.array([prev_3d[i] for i, _ in pairs])
        dst = np.array([cur_3d[j] for _, j in pairs])
        try:
            new_R, _, _ = kabsch(src, dst)
        except Exception:
            break

        diff = np.linalg.norm(new_R - current_R)
        current_R = new_R
        if diff < 1e-4:
            break

    # 最终匹配用修正后的 R
    return _match_with_R(current_R, prev_3d, cur_3d, max_displacement)


def track_dots_across_frames(frames_3d, valid_fi, max_displacement=None,
                              rot_per_frame=None):
    """
    逐帧匹配。提供 rot_per_frame 时用 ICP 旋转修正匹配, 否则回退纯 3D 邻近匹配。
    返回 (frame_labels, next_gid, matched_gids)。
    matched_gids[fi] = 从上一帧成功继承的 gid 集合。
    """
    if max_displacement is None: max_displacement = MATCH_MAX_DISP
    frame_labels = {}
    matched_gids = {}
    next_gid = 0

    for idx, fi in enumerate(valid_fi):
        dots_3d = frames_3d[fi]
        local_label = {}
        local_matched = set()

        if idx == 0:
            for i in range(len(dots_3d)):
                local_label[i] = next_gid
                next_gid += 1
        else:
            prev_fi = valid_fi[idx - 1]
            prev_3d = frames_3d[prev_fi]
            prev_label = frame_labels[prev_fi]

            if rot_per_frame is not None:
                R, _ = cv2.Rodrigues(rot_per_frame)
                matches = match_by_rotation_icp(prev_3d, dots_3d, R, max_displacement)
            else:
                matches = match_by_3d_proximity(prev_3d, dots_3d, max_displacement)

            for pi, ci in matches:
                if pi in prev_label:
                    gid = prev_label[pi]
                    local_label[ci] = gid
                    local_matched.add(gid)
            for i in range(len(dots_3d)):
                if i not in local_label:
                    local_label[i] = next_gid
                    next_gid += 1

        frame_labels[fi] = local_label
        matched_gids[fi] = local_matched

    return frame_labels, next_gid, matched_gids
