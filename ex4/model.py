import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class upd_GELU(nn.Module):
    def __init__(self):
        super(upd_GELU, self).__init__()
        self.constant_param = nn.Parameter(torch.Tensor([1.702]))
        self.sig = nn.Sigmoid()
    
    def forward(self, input: Tensor) -> Tensor:
        outval = torch.mul(self.sig(torch.mul(self.constant_param, input)), input)
        return outval

class KerasMaxPool1d(nn.Module):
    def __init__(
        self,
        pool_size=2,
        padding="valid",
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        """Hard coded to only be compatible with kernel size and stride of 2."""
        super().__init__()
        self.padding = padding
        _padding = 0
        if pool_size != 2:
            raise NotImplementedError("MaxPool1D with kernel size other than 2.")
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size,
            padding=_padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
    
    def forward(self, x):
        # (bch)
        if self.padding == "same" and x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1), value=-float("inf"))
        return self.pool(x)

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return self.module(x) + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=1, padding='same',
                stride=1, dilation_rate=1, pool_size=1, dropout=0, bn_momentum=0.1):
        super().__init__()
        block = nn.ModuleList()
        block.append(upd_GELU())
        block.append(nn.Conv1d(in_channels=in_channels,
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=int(round(dilation_rate)),
                            bias=False))
        block.append(nn.BatchNorm1d(filters, momentum=bn_momentum, affine=True))
        if dropout > 0:
            block.append(nn.Dropout(p=dropout))
        if pool_size > 1:
            block.append(KerasMaxPool1d(pool_size=pool_size, padding=padding))
        self.block = nn.Sequential(*block)
        self.out_channels = filters
    
    def forward(self, x):
        return self.block(x)

class ConvTower(nn.Module):
    def __init__(
        self,
        in_channels,
        filters_init,
        filters_end=None,
        filters_mult=None,
        divisible_by=1,
        repeat=2,
        **kwargs,
    ):
        super().__init__()
        
        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)
        
        # determine multiplier
        if filters_mult is None:
            assert filters_end is not None
            filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))
        
        rep_filters = filters_init
        in_channels = in_channels
        tower = nn.ModuleList()
        for _ in range(repeat):
            tower.append(
                ConvBlock(
                    in_channels=in_channels, filters=_round(rep_filters), **kwargs
                )
            )
            in_channels = _round(rep_filters)
            rep_filters *= filters_mult
        
        self.tower = nn.Sequential(*tower)
        self.out_channels = in_channels
    
    def forward(self, x):
        return self.tower(x)

class DilatedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size=3,
        rate_mult=2,
        dropout=0,
        repeat=1,
        **kwargs,
    ):
        super().__init__()
        dilation_rate = 1  # 初始化为1，后面累乘
        in_channels = in_channels
        block = nn.ModuleList()
        for _ in range(repeat):
            inner_block = nn.ModuleList()
            
            inner_block.append(
                ConvBlock(
                    in_channels=in_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=int(np.round(dilation_rate)),
                    **kwargs,
                )
            )
            
            inner_block.append(
                ConvBlock(
                    in_channels=filters,
                    filters=in_channels,
                    dropout=dropout,
                    **kwargs,
                )
            )
            
            block.append(Residual(nn.Sequential(*inner_block)))
            
            dilation_rate *= rate_mult
            dilation_rate = np.round(dilation_rate)
        self.block = nn.Sequential(*block)
        self.out_channels = in_channels
    
    def forward(self, x):
        return self.block(x)

class BasenjiFinal(nn.Module):
    def __init__(
        self, in_features, units=1, activation='linear', **kwargs):
        super().__init__()
        block = nn.ModuleList()
        block.append(Rearrange('b ... -> b (...)'))
        
        # Rearrange('b ... -> b (...)') 代表将batch size维度保留，剩下的维度变成一维
        # 这里相当于flatten
        
        block.append(nn.Linear(in_features=in_features, out_features=units))
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)

class BasenjiModel(nn.Module):
    def __init__(self, conv1_filters=8, conv1_ks=15,
                # 第一个conv block参数
                conv1_pad=7, conv1_pool=2, conv1_pdrop=0.4, conv1_bn_momentum=0.1,
                # conv tower参数
                convt_filters_init=16, filters_end=32, convt_repeat=2, convt_ks=5, convt_pool=2,
                # dilres block参数
                dil_in_channels=32, dil_filters=16, dil_ks=3, rate_mult=2, dil_pdrop=0.3, dil_repeat=2,
                conv2_in_channels=32, conv2_filters=32,  # 第二个conv block参数
                conv3_in_channels=32, conv3_filters=1,  # 第三个conv block(1*1 conv)参数
                final_in_features=int(3000/(2**3))  # final block参数，2**3表示经过3次pooling
                ):
        super().__init__()
        block = nn.ModuleList()
        block.append(ConvBlock(in_channels=4,
                             filters=conv1_filters,
                             kernel_size=conv1_ks,
                             padding=conv1_pad,
                             pool_size=conv1_pool,
                             dropout=conv1_pdrop,
                             bn_momentum=conv1_bn_momentum))
        
        block.append(ConvTower(in_channels=conv1_filters,
                             filters_init=convt_filters_init,
                             filters_end=filters_end,
                             divisible_by=1,
                             repeat=convt_repeat,
                             kernel_size=convt_ks,
                             pool_size=convt_pool,
                             ))
        
        block.append(DilatedResidual(
            in_channels=dil_in_channels,
            filters=dil_filters,
            kernel_size=dil_ks,
            rate_mult=rate_mult,
            dropout=dil_pdrop,
            repeat=dil_repeat,))
        
        block.append(ConvBlock(in_channels=conv2_in_channels,
                             filters=conv2_filters,
                             kernel_size=1))
        block.append(ConvBlock(in_channels=conv3_in_channels,
                             filters=conv3_filters,
                             kernel_size=1))
        block.append(BasenjiFinal(final_in_features))
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)

# 辅助函数 - 定义自定义数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input, label):
        inputs = torch.tensor(input, dtype=torch.int8)
        labels = torch.tensor(label, dtype=torch.float32)
        self.inputs = inputs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    # 仅用于测试模型结构
    from torchsummary import summary
    
    model = BasenjiModel()
    # 假设输入形状为 [batch_size, 4, 3000]
    summary(model, input_size=[(4, 3000)], batch_size=32, device="cpu")