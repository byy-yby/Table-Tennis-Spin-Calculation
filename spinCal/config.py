"""全局常量 — 零硬编码路径"""
import torch
import numpy as np

# 相机内参
MTX = np.array([[1914.91362, 0., 802.626903],
                 [0., 1906.44992, 542.879886],
                 [0., 0., 1.]], dtype=np.float64)
DIST = np.array([-0.17027915, 0.220013, 0.00154302, -0.00201744, 1.28603106],
                dtype=np.float64)
BALL_RADIUS_MM = 20.0
CALIB_IMAGE_HEIGHT = 1080

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 黑点检测
CNN_HMAP_THRESH = 0.3
MAX_DOT_EDGE_RATIO = 0.85  # 只取球心85%半径内的点, 边缘投影误差大
HOUGH_DP = 1.2
HOUGH_PARAM1 = 40
HOUGH_PARAM2 = 25

# 匹配
MATCH_MAX_DISP = 0.3        # Stage 1 粗匹配 (需覆盖每帧位移: 2000RPM@916fps≈0.226弦长)
ICP_MATCH_DISP = 0.12       # Stage 3 ICP 紧阈值 (有旋转约束)
MIN_COMMON_DOTS = 3         # 帧对最低公共点数 (2点Kabsch欠定, 轴随机)

# 轨迹一致性
TRAJ_RMS_THRESH = 0.10      # 点轨迹 RMS > 此值 → 剔除 (单位球面弦长, ≈5.7°)

# 显示
DOT_COLORS = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255),
              (0, 165, 255), (255, 255, 0), (128, 255, 0), (255, 0, 128)]
