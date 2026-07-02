"""DotNet CNN 模型定义 + 加载"""
import torch, torch.nn as nn


class DotNet(nn.Module):
    """全卷积热图回归网络 — 无池化, 保持分辨率 (同 SpinDOE 设计)"""
    def __init__(self):
        super().__init__()
        channels = [3, 32, 32, 64, 64, 128, 128, 64, 64, 32, 32]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 3, padding=1))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.body = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out(self.body(x))


def load_model(checkpoint_path, device=None):
    """加载训练好的 DotNet 模型"""
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DotNet().to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    return model
