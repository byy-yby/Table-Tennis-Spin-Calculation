"""BallNet — CoordConv + Sigmoid 回归 (cx/60, cy/60, r/30 ∈ [0,1])"""
import torch, torch.nn as nn, numpy as np


class BallNet(nn.Module):
    """
    CoordConv: 在 RGB 输入上拼接 X/Y 坐标通道, 让模型知道每个像素的空间位置。
    输出: 3 个值 ∈ [0,1] (Sigmoid), 对应 cx/60, cy/60, r/30。
    """
    def __init__(self):
        super().__init__()
        channels = [5, 24, 24, 48, 48, 64, 48, 32, 32, 16]  # 5 = 3 RGB + 2 coord
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 3, padding=1))
            if i < len(channels) - 2:
                layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(nn.Dropout2d(0.35))
        self.body = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4), nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 3),
            nn.Sigmoid()  # 输出严格 ∈ [0,1]
        )

        # 注册坐标网格 (不参与训练)
        self.register_buffer('coord_grid', self._make_coord_grid(60))

    @staticmethod
    def _make_coord_grid(size):
        ys = torch.linspace(-1, 1, size)
        xs = torch.linspace(-1, 1, size)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx, gy], dim=0)  # (2, 60, 60)

    def forward(self, x):
        # 拼接坐标通道
        B = x.shape[0]
        coords = self.coord_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, 60, 60)
        x = torch.cat([x, coords], dim=1)  # (B, 5, 60, 60)
        feats = self.body(x)
        return self.head(feats)  # (B, 3) ∈ [0,1]


def pred_to_absolute(pred, img_w=60, img_h=60):
    """Sigmoid 输出 [0,1] → 像素坐标: cx = out[0]*60, cy = out[1]*60, r = out[2]*30"""
    arr = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
    return arr[0] * img_w, arr[1] * img_h, arr[2] * 30.0


def load_ballnet(checkpoint_path, device=None):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallNet().to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    return model

