import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的ResidualBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x))) + self.shortcut(x)
    
# 加权通道重要性
class Channel(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(Channel, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels,
                             out_channels=internal_neurons,
                             kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons,
                             out_channels=input_channels,
                             kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # 平均池化分支
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        # 最大池化分支
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        return x

# 深度卷积
class DwConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DwConv, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2),
            groups=in_channels
        )
        
    def forward(self, x):
        return self.dwconv(x)

# 定义改进后的网络
class improve_Net(nn.Module):
    def __init__(self, keep_prob=0.7):
        super(improve_Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            ResBlock(32, 64),
            nn.MaxPool2d(2), 
            DwConv(64, (1, 3)), 
            DwConv(64, (3, 1)), 
            ResBlock(64, 128),  
            nn.MaxPool2d(2),
            #DwConv(128, (1, 3)),
            #DwConv(128, (3, 1)), 
            nn.AdaptiveAvgPool2d(1)
        )
        # 通道注意力模块
        self.ca = Channel(input_channels=128, internal_neurons=128 // 4)
        # 全连接层和Dropout
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(1 - keep_prob)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)  # 通过卷积层和注意力层
        x = self.ca(x) * x  # 通道注意力加权

        x = torch.flatten(x, 1)  # 展平，保留batch维度
        x = self.fc1(x)  # 全连接
        x = self.relu(x)  # 激活
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # 最终分类
        x = self.softmax(x)  # Softmax输出
        return x