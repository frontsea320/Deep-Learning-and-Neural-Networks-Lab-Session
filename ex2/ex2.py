import torch
from torch import nn

# 定义 CNN 网络
class Net(nn.Module):
    def __init__(self, keep_prob=0.7):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=7 * 7 * 64, out_features=1024),
            nn.ReLU(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(in_features=1024, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.model(input)