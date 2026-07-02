# Ping-Pong Ball Spin Measurement

基于 CNN 热图回归 + 3D 球面匹配 + Kabsch 刚性旋转计算的乒乓球转速自动测量系统。

---

## 测量原理

### 1. 球体定位与裁剪

从高速摄像机视频中，通过背景差分找到球出现的帧区间，用霍夫圆检测确定球心位置和像素半径，裁剪出 60×60 的球体 ROI。

### 2. CNN 黑点检测

使用预训练的 DotNet（全卷积热图回归网络）在 60×60 球体图像上预测黑点热图。每个黑点对应热图上的一个高斯峰值。通过非极大值抑制提取 2D 像素坐标。

### 3. 2D → 3D 投影

利用标定好的相机内参（针孔模型），通过射线-球面求交将每个 2D 黑点坐标投影到球体表面的 3D 位置（单位球面，球心为原点）。

### 4. 跨帧匹配

相邻帧之间，同一物理黑点在单位球面上的 3D 位置变化很小。用匈牙利算法在 3D 空间中做贪心最近邻匹配，为每个黑点分配全局唯一 ID。

### 5. Kabsch 旋转计算

对 3 帧滑动窗口内同时存在的黑点（>=3 个），用 Kabsch 算法计算帧间的最优旋转矩阵。Kabsch 不减质心（点已是球心相对向量），确保旋转轴穿过球心。通过全局轴对齐 + 中值滤波去除异常值，最终输出：

- **RPM** (转/分钟) 和 **RPS** (转/秒)
- **旋转轴方向** (3D 单位向量)
- **旋转类型** (上旋/下旋/侧旋/陀螺旋)
- **旋转方向** (顺时针/逆时针)

---

## 目录结构

```
BDprogram/
├── spinCal/              # 旋转测量核心包
│   ├── main.py           # CLI 入口
│   ├── config.py         # 相机参数 & 阈值常量
│   ├── geometry.py       # 3D 几何 (射线求交, Kabsch)
│   ├── model.py          # DotNet CNN 模型
│   ├── detection.py      # 单帧处理 (霍夫圆 + CNN + 2D→3D)
│   ├── matching.py       # 3D 邻近匹配 & ID 追踪
│   ├── rotation.py       # 3帧窗口 Kabsch + 中值滤波
│   ├── video.py          # AVI → 裁剪帧
│   └── viz.py            # 交互式可视化 & 3D 球面
│
├── getData/              # 数据生产
│   ├── extract.py        # 批量从 AVI 提取球体裁剪帧
│   ├── label.py          # 手动标注黑点位置
│   └── augment.py        # 数据增强 (翻转/旋转/亮度/噪声)
│
├── dotnet/               # 模型训练
│   ├── model.py           # DotNet 网络定义
│   ├── dataset.py         # 热图回归数据集
│   └── train.py           # 训练脚本
│
├── spinCal2.py            # 旧版单文件 (功能同 spinCal 包)
├── dotnet_vis.py          # CNN 检测可视化工具
├── train_dotnet.py        # 旧版训练脚本
└── README.md              # 本文件
```

---

## 坐标系约定

本系统全程使用 **OpenCV 原生相机坐标系**，不做任何翻转：

```
+X : 相机画面向右
+Y : 相机画面向下
+Z : 相机光轴向前 (远离相机)
```

3D 球面可视化中，旋转轴青色箭头指向 +Z 表示逆时针 (CCW)，指向 -Z 表示顺时针 (CW)（从相机方向观察）。

---

## 使用流程

### 环境要求

```bash
conda create -n spindoe10 python=3.10 -y
conda activate spindoe10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python scikit-learn scipy tqdm matplotlib pandas ultralytics
```

### 一步测量 (推荐)

```bash
# 直接从 AVI 视频测量旋转
python -m spinCal.main D:/highspeed/test10.avi

# 从图片文件夹测量
python -m spinCal.main dataset/data_data1 --model dataset/dotnet.pt

# 跳过可视化
python -m spinCal.main D:/highspeed/test10.avi --no-viz
```

### 自定义训练流程

如果你需要用自己标注的数据训练模型：

```bash
# 1. 从视频提取帧
python getData/extract.py D:/highspeed/data

# 2. 手动标黑点 (左键加点, 右键撤销, ENTER下一帧)
python getData/label.py dataset

# 3. 数据增强
python getData/augment.py dataset

# 4. 训练模型
python dotnet/train.py dataset --epochs 100

# 5. 使用新模型测量
python -m spinCal.main D:/highspeed/test10.avi --model dataset/dotnet.pt
```

---

## 可调参数

在 `spinCal/config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MATCH_MAX_DISP` | 0.5 | 3D 匹配最大位移 (单位球面弧长) |
| `CNN_HMAP_THRESH` | 0.3 | CNN 热图检出阈值 |
| `HOUGH_PARAM1` | 40 | 霍夫圆 Canny 阈值 |
| `HOUGH_PARAM2` | 25 | 霍夫圆累加器阈值 |

也可命令行临时覆盖：`--match-disp 0.3`

---

## 依赖

- Python >= 3.10
- PyTorch + CUDA
- OpenCV
- NumPy, SciPy
- Matplotlib
- Ultralytics (可选, 仅旧版脚本使用)

---

## 参考

- Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors"
- SpinDOE: Gossard et al., "A ball spin estimation method for table tennis robot", arXiv:2303.03879
