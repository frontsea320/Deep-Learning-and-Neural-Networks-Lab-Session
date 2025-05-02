# quaternion_layers.py
import torch
import torch.nn as nn

class QuaternionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        """
        四元数一维卷积层，支持空洞（dilation）卷积。
        参数:
          in_channels, out_channels: 必须为4的倍数，每4个通道代表一个四元数。
          kernel_size, stride, padding, dilation: 与普通卷积相同。
        """
        super(QuaternionConv1d, self).__init__()
        self.r_conv = nn.Conv1d(in_channels // 4, out_channels // 4, kernel_size,
                                stride, padding, dilation, bias=bias)
        self.i_conv = nn.Conv1d(in_channels // 4, out_channels // 4, kernel_size,
                                stride, padding, dilation, bias=bias)
        self.j_conv = nn.Conv1d(in_channels // 4, out_channels // 4, kernel_size,
                                stride, padding, dilation, bias=bias)
        self.k_conv = nn.Conv1d(in_channels // 4, out_channels // 4, kernel_size,
                                stride, padding, dilation, bias=bias)

    def forward(self, x):
        # x 的 shape: (batch, in_channels, L)
        r, i, j, k = torch.chunk(x, 4, dim=1)
        r_out = self.r_conv(r) - self.i_conv(i) - self.j_conv(j) - self.k_conv(k)
        i_out = self.r_conv(i) + self.i_conv(r) + self.j_conv(k) - self.k_conv(j)
        j_out = self.r_conv(j) - self.i_conv(k) + self.j_conv(r) + self.k_conv(i)
        k_out = self.r_conv(k) + self.i_conv(j) - self.j_conv(i) + self.k_conv(r)
        return torch.cat([r_out, i_out, j_out, k_out], dim=1)


class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        四元数全连接层
        参数:
           in_features, out_features: 必须为4的倍数，每4个数字表示一个四元数。
        """
        super(QuaternionLinear, self).__init__()
        self.r_linear = nn.Linear(in_features // 4, out_features // 4, bias=bias)
        self.i_linear = nn.Linear(in_features // 4, out_features // 4, bias=bias)
        self.j_linear = nn.Linear(in_features // 4, out_features // 4, bias=bias)
        self.k_linear = nn.Linear(in_features // 4, out_features // 4, bias=bias)

    def forward(self, x):
        # x 的 shape: (batch, in_features)
        r, i, j, k = torch.chunk(x, 4, dim=1)
        r_out = self.r_linear(r) - self.i_linear(i) - self.j_linear(j) - self.k_linear(k)
        i_out = self.r_linear(i) + self.i_linear(r) + self.j_linear(k) - self.k_linear(j)
        j_out = self.r_linear(j) - self.i_linear(k) + self.j_linear(r) + self.k_linear(i)
        k_out = self.r_linear(k) + self.i_linear(j) - self.j_linear(i) + self.k_linear(r)
        return torch.cat([r_out, i_out, j_out, k_out], dim=1)