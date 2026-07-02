"""3D 邻近匹配 + 跨帧 ID 追踪"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import MATCH_MAX_DISP


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
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < max_displacement:
            matches.append((i, j))
    return matches


def track_dots_across_frames(frames_3d, valid_fi, max_displacement=None):
    """逐帧匹配, 返回 frame_labels 和全局 ID 信息"""
    if max_displacement is None: max_displacement = MATCH_MAX_DISP
    frame_labels = {}
    next_gid = 0

    for idx, fi in enumerate(valid_fi):
        dots_3d = frames_3d[fi]
        local_label = {}
        if idx == 0:
            for i in range(len(dots_3d)):
                local_label[i] = next_gid
                next_gid += 1
        else:
            prev_fi = valid_fi[idx - 1]
            prev_3d = frames_3d[prev_fi]
            prev_label = frame_labels[prev_fi]
            matches = match_by_3d_proximity(prev_3d, dots_3d, max_displacement)
            matched_cur = set()
            for pi, ci in matches:
                if pi in prev_label:
                    gid = prev_label[pi]
                    local_label[ci] = gid
                    matched_cur.add(ci)
            for i in range(len(dots_3d)):
                if i not in matched_cur:
                    local_label[i] = next_gid
                    next_gid += 1
        frame_labels[fi] = local_label

    return frame_labels, next_gid
