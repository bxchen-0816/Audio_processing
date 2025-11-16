import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 通道注意力"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.avg(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w

class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, use_se: bool = True, se_reduction: int = 8, dropout: float = 0.3):
        super().__init__()
        # block1: 1 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.se1   = SEBlock(32, se_reduction) if use_se else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)

        # block2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.se2   = SEBlock(64, se_reduction) if use_se else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)

        # block3: 64 -> 128  (无池化)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.se3 = SEBlock(128, se_reduction) if use_se else nn.Identity()

        # head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.pool1(self.se1(self.conv1(x)))
        x = self.pool2(self.se2(self.conv2(x)))
        x = self.se3(self.conv3(x))
        return self.head(x)