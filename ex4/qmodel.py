# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from qlayer import QuaternionConv1d, QuaternionLinear

class DilationResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        """
        四元数空洞残差块
        参数:
          channels: 输入和输出通道数（必须为4的倍数）。
          kernel_size: 卷积核大小。
          dilation: 空洞扩张率。
        """
        super(DilationResidualBlock, self).__init__()
        # 计算 padding 保证输出尺寸不变
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = QuaternionConv1d(channels, channels, kernel_size,
                                      padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        # 第二层采用标准卷积，dilation=1，padding保证尺寸相同
        self.conv2 = QuaternionConv1d(channels, channels, kernel_size,
                                      padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class QuaternionBasenjiModel(nn.Module):
    def __init__(self, input_channels, seq_length):
        """
        模型用于预测基因表达量（TPM），基于四元数空洞残差卷积网络。
        参数:
          input_channels: 输入通道数（one-hot 编码时为4）。
          seq_length: DNA 序列长度（例如 3000）。
        """
        super(QuaternionBasenjiModel, self).__init__()
        # 初始卷积块：将 4 个通道扩展到 64 个通道（64 必须为4的倍数）
        self.initial_conv = QuaternionConv1d(input_channels, 64, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(64)
        
        # 6 个空洞残差块，空洞率依次为 1,2,4,8,16,32
        self.res_blocks = nn.ModuleList([
            DilationResidualBlock(64, kernel_size=3, dilation=2**i) for i in range(6)
        ])
        
        # 全局平均池化（对序列长度维度做全局池化）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接部分：先通过一个四元数全连接层，输出 64 个四元数（即64*4 个数），再计算四元数模
        self.fc_q = QuaternionLinear(64, 64 * 4)  
        # 最后接一个普通全连接层，将 64 维特征映射到 1 个输出（TPM）
        self.fc_final = nn.Linear(64, 1)

    def forward(self, x):
        # x 的 shape: (batch, input_channels, seq_length)
        x = F.relu(self.initial_bn(self.initial_conv(x)))  # (batch, 64, L)
        for block in self.res_blocks:
            x = block(x)  # 保持 (batch, 64, L)
        x = self.global_pool(x)  # (batch, 64, 1)
        x = x.squeeze(2)       # (batch, 64)
        # 全连接层（四元数全连接）
        x = self.fc_q(x)       # (batch, 64*4)
        # 重塑为 (batch, 64, 4)，每个样本 64 个四元数
        x = x.view(x.size(0), 64, 4)
        # 计算四元数模：sqrt(r^2 + i^2 + j^2 + k^2)，输出 (batch, 64)
        x_norm = torch.sqrt((x ** 2).sum(dim=2) + 1e-6)
        out = self.fc_final(x_norm)  # (batch, 1)
        return out