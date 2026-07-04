"""单帧处理: BallNet球检测 + CNN黑点 + 2D→3D"""
import warnings, cv2, torch, numpy as np
from scipy.ndimage import maximum_filter
from .config import CNN_HMAP_THRESH, BALL_RADIUS_MM, MAX_DOT_EDGE_RATIO, DEVICE
from .geometry import ball_center_to_3d, ray_sphere_intersection

warnings.filterwarnings("ignore", message=".*weights_only.*")


class BallDetector:
    """BallNet 球心+半径回归器 (替代霍夫圆)"""
    def __init__(self, model_path, device=None):
        if device is None: device = DEVICE
        self.device = device
        # 延迟导入避免循环依赖
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "ballnet"))
        from model import BallNet
        self.model = BallNet().to(device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=device,
                                                weights_only=False))
        self.model.eval()

    def detect(self, img):
        """返回 (cx, cy, r) 或 None。img 是 60×60 BGR"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw = self.model(tensor)  # (1, 3) ∈ [0,1] via Sigmoid
        pred = raw[0].cpu().numpy()

        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "ballnet"))
        from model import pred_to_absolute
        cx, cy, r = pred_to_absolute(raw[0])

        if not hasattr(self, '_logged'):
            self._logged = True
            print(f"[BallNet] raw=({pred[0]:.4f},{pred[1]:.4f},{pred[2]:.4f}) "
                  f"→ ({cx:.1f},{cy:.1f},{r:.1f})")

        if r < 5:
            return None
        return float(cx), float(cy), float(r)


def process_image(img, model, ball_detector=None, fixed_ball=None):
    """处理单张图: 球检测 → CNN黑点 → 2D→3D。fixed_ball=(cx,cy,r) 跳过检测。"""
    h, w = img.shape[:2]

    if fixed_ball is not None:
        bcx, bcy, br = fixed_ball
    elif ball_detector is not None:
        ball = ball_detector.detect(img)
        if ball is not None:
            bcx, bcy, br = ball
        else:
            return [], [], None
    else:
        return [], [], None

    # CNN 黑点检测
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()
    peaks = (pred == maximum_filter(pred, size=5)) & (pred > CNN_HMAP_THRESH)
    ys, xs = np.where(peaks)
    dots_2d_raw = list(zip(xs, ys))

    # 过滤球边缘点: 距球心 > 85% 半径的点投影误差大, 丢弃
    dots_2d = []
    max_dist = MAX_DOT_EDGE_RATIO * br
    for dx, dy in dots_2d_raw:
        if np.hypot(dx - bcx, dy - bcy) <= max_dist:
            dots_2d.append((dx, dy))

    # 2D→3D
    bc_3d = ball_center_to_3d(bcx, bcy, br)
    dots_3d = []
    for dx, dy in dots_2d:
        p3d = ray_sphere_intersection(dx, dy, bc_3d)
        if p3d is not None:
            v = (p3d - bc_3d) / BALL_RADIUS_MM
            dots_3d.append(np.array([v[0], v[1], v[2]]))

    return dots_2d, dots_3d, (bcx, bcy, br, bc_3d)
