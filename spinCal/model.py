"""DotNet CNN 模型定义 + 加载"""
import warnings, torch, torch.nn as nn
warnings.filterwarnings("ignore", message=".*weights_only.*")


class DotNet(nn.Module):
    """全卷积热图回归网络 — 无池化, 保持分辨率"""
    def __init__(self):
        super().__init__()
        channels = [3, 24, 24, 48, 48, 64, 48, 32, 32, 16]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 3, padding=1))
            if i < len(channels) - 2:
                layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(nn.Dropout2d(0.35))
        self.body = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(8, 1, 1),
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
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device,
                                       weights_only=True))
    model.eval()
    return model
