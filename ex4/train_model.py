import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from torchsummary import summary
from sklearn.model_selection import train_test_split

# One-hot编码函数
def one_hot_encode_along_channel_axis(sequence):
    # 将DNA序列（ACGT）转换为one-hot编码
    sequence = sequence.upper()
    seq_len = len(sequence)
    one_hot = np.zeros((4, seq_len), dtype=np.int8)
    
    for i in range(seq_len):
        if sequence[i] == 'A':
            one_hot[0, i] = 1
        elif sequence[i] == 'C':
            one_hot[1, i] = 1
        elif sequence[i] == 'G':
            one_hot[2, i] = 1
        elif sequence[i] == 'T':
            one_hot[3, i] = 1
    
    return one_hot

# 数据集加载和预处理
# 为了统一，这里提前为大家划分好了训练集，验证集和测试集，见df['dataset']列
# 导入划分好的数据集, 并对DNA序列one-hot编码处理
df = pd.read_excel('./dataset.xlsx')
print('genes number:', df.shape[0])

df_train = df[df['dataset'] == 'train']
y_train = np.log2(df_train['TPM'].values + 1)
train_data = np.array([one_hot_encode_along_channel_axis(i.strip()) for i in df_train['sequence'].values])

df_valid = df[df['dataset'] == 'valid']
y_valid = np.log2(df_valid['TPM'].values + 1)
valid_data = np.array([one_hot_encode_along_channel_axis(i.strip()) for i in df_valid['sequence'].values])

df_test = df[df['dataset'] == 'test']
y_test = np.log2(df_test['TPM'].values + 1)
test_data = np.array([one_hot_encode_along_channel_axis(i.strip()) for i in df_test['sequence'].values])

# 定义模型所需的组件
class upd_GELU(nn.Module):
    def __init__(self):
        super(upd_GELU, self).__init__()
        self.constant_param = nn.Parameter(torch.Tensor([1.702]))
        self.sig = nn.Sigmoid()
    
    def forward(self, input: Tensor) -> Tensor:
        outval = torch.mul(self.sig(torch.mul(self.constant_param, input)), input)
        return outval

# 由于torch的polling操作没有padding='same', 因此重新定义了maxpooling类
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

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, input, label):
        inputs = torch.tensor(input, dtype=torch.int8)
        labels = torch.tensor(label, dtype=torch.float32)
        self.inputs = inputs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 10
lr = 1e-3

# valid data可以用于早停，这里没有加入早停机制，你们可以自己尝试加入
train_dataset = MyDataset(train_data, y_train)
trainDataLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset = MyDataset(valid_data, y_valid)
validDataLoader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)
test_dataset = MyDataset(test_data, y_test)
testDataLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

# 配置设备和训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

net = BasenjiModel()
summary(net, input_size=[(4, 3000)], batch_size=BATCH_SIZE, device="cpu")
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
lossF = nn.MSELoss()
print(net.to(device))

# 存储训练过程
history = {'Valid Loss': [], 'Valid pcc': []}
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')
    net.train(True)
    epoch_loss_all = 0
    for step, (inputs, labels) in enumerate(processBar):
        # 将序列和标签传输进device中
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 清空模型的梯度
        net.zero_grad()
        # 对模型进行前向推理
        outputs = net(inputs)
        # 计算本轮推理的Loss值
        loss = lossF(outputs.reshape(labels.shape), labels)
        epoch_loss_all += loss * len(labels)
        # 进行反向传播求出模型参数的梯度
        loss.backward()
        # 使用迭代器更新模型权重
        optimizer.step()
        # 将本step结果进行可视化处理
        processBar.set_description("[%d/%d] Loss: %.4f" %
                                (epoch, EPOCHS, loss.item()))
        if step == len(processBar) - 1:
            valid_totalLoss = 0
            y_valid_pred_all = []
            y_valid_true_all = []
            train_loss = epoch_loss_all / len(train_data)
            # 关闭模型的训练状态
            net.train(False)
            with torch.no_grad():
                # 对验证集的DataLoader进行迭代
                for x_valid, y_valid in validDataLoader:
                    x_valid = x_valid.to(device)
                    y_valid = y_valid.to(device)
                    y_valid_pred = net(x_valid)
                    y_valid_pred_all.extend(y_valid_pred.flatten().tolist())
                    y_valid_true_all.extend(y_valid.tolist())
                    valid_batch_loss = lossF(y_valid_pred.reshape(y_valid.shape), y_valid)
                    valid_totalLoss += valid_batch_loss * len(y_valid)
                validLoss = valid_totalLoss / len(valid_data)
                pcc_valid, _ = pearsonr(y_valid_pred_all, y_valid_true_all)
                history['Valid Loss'].append(validLoss.item())
                history['Valid pcc'].append(pcc_valid)
                processBar.set_description("[%d/%d] Train Loss: %.4f, Valid Loss: %.4f, Valid pcc: %.4f" %
                                        (epoch, EPOCHS, train_loss.item(), validLoss.item(), pcc_valid))
    processBar.close()

# 在测试集上评估模型性能
# 计算测试集上的pcc和R2，并画测试集上真实值和预测值之间的散点图
net.train(False)
y_test_pred_all = []
y_test_true_all = []
with torch.no_grad():
    # 对测试集的DataLoader进行迭代
    for x_test, y_test in testDataLoader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_test_pred = net(x_test)
        y_test_pred_all.extend(y_test_pred.flatten().tolist())
        y_test_true_all.extend(y_test.tolist())

# 计算测试集上的pcc，并画测试集上真实值和预测值之间的散点图
pcc, _ = pearsonr(y_test_pred_all, y_test_true_all)
print('pcc:', pcc)
plt.xlabel('y test predict')
plt.ylabel('y test true')
plt.scatter(y_test_pred_all, y_test_true_all, s=0.1)
plt.show()
plt.savefig('ori_plot.png')  # 替换为你想要保存的路径
