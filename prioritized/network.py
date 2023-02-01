import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.Flatten()
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, outputs)
        )

    def forward(self, x):
        x = x.to(device)
        x = self.cnn_stack(x)
        x = self.linear_stack(x)
        return x
