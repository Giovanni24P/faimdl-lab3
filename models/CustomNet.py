import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), # Out: [B, 64, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Out: [B, 64, 112, 112]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Out: [B, 128, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Out: [B, 128, 56, 56]

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Out: [B, 256, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Out: [B, 256, 28, 28]

            nn.Flatten(), 
            nn.Linear(256 * 28 * 28, 200)
        )

    def forward(self, x):
        x = self.sequential(x)
        return x