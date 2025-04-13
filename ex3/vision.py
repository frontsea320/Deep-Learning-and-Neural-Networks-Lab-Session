import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot
import os
import math
from improve_model import ConvRNN, ConvolutionalWordEmbedding, TextConverter
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

# 设置参数
vocab_size = 5001  # 根据你的模型参数调整
embed_dim = 256    # 嵌入维度
hidden_size = 512  # 隐藏层大小
num_layers = 2     # RNN层数
batch_size = 8     # 批次大小用于可视化
seq_length = 16    # 序列长度用于可视化

# 创建模型
model = ConvRNN(
    num_classes=vocab_size,
    embed_dim=embed_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=0.5
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 打印模型结构
print("模型基本结构:")
print(model)

# 使用torchsummary打印详细模型参数量
# 注意: 需要安装torchsummary (pip install torchsummary)
print("\n模型参数详细统计:")
try:
    summary(model, input_size=(seq_length,), dtypes=[torch.long], device=device)
except Exception as e:
    print(f"无法使用torchsummary: {e}")
    print("尝试手动展示模型参数...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 手动打印每层参数
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} 参数")

# 创建一个示例输入用于可视化
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

# 使用torchviz生成计算图可视化 (需要安装graphviz和torchviz)
print("\n生成计算图可视化...")
try:
    output, _ = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    # 保存图形
    dot.format = 'png'
    dot.render('model_architecture', cleanup=True)
    print(f"模型计算图已保存为 'model_architecture.png'")
except Exception as e:
    print(f"无法生成计算图: {e}")

# 模型层次结构的文本可视化
def print_model_hierarchy(model, indent=0):
    """打印模型的层次结构，带有缩进"""
    first_child = True
    for name, module in model.named_children():
        connector = "└─" if first_child else "├─"
        print("│  " * indent + connector + name)
        if list(module.children()):
            print_model_hierarchy(module, indent + 1)
        first_child = False

print("\n模型层次结构:")
print_model_hierarchy(model)

# 自定义函数，创建网络结构图
def create_network_diagram():
    """创建网络结构的可视化图"""
    try:
        from graphviz import Digraph
        
        dot = Digraph('ConvRNN', comment='卷积增强型RNN模型结构')
        
        # 添加节点
        dot.node('input', '输入序列\n(batch_size, seq_length)', shape='ellipse')
        dot.node('embedding', '词嵌入层\n(词汇量, embed_dim)', shape='box')
        dot.node('conv_embed', '卷积嵌入层\n(卷积核: 3x1)', shape='box')
        dot.node('rnn', f'RNN层\n({num_layers}层, hidden_size={hidden_size})', shape='box')
        dot.node('linear', '线性投影层\n(hidden_size, vocab_size)', shape='box')
        dot.node('output', '输出\n(batch_size * seq_length, vocab_size)', shape='ellipse')
        
        # 添加边
        dot.edge('input', 'embedding')
        dot.edge('embedding', 'conv_embed')
        dot.edge('conv_embed', 'rnn')
        dot.edge('rnn', 'linear')
        dot.edge('linear', 'output')
        
        # 保存图形
        dot.format = 'png'
        dot.render('network_diagram', cleanup=True)
        print(f"网络结构图已保存为 'network_diagram.png'")
    except Exception as e:
        print(f"无法创建网络结构图: {e}")

print("\n创建网络结构图...")
create_network_diagram()

# 创建更详细的ConvolutionalWordEmbedding可视化
def create_conv_embedding_diagram():
    """创建卷积嵌入层的详细可视化图"""
    try:
        from graphviz import Digraph
        
        dot = Digraph('ConvEmbedding', comment='卷积嵌入层结构')
        
        # 添加节点
        dot.node('input_embed', '词嵌入输入\n(batch, seq_len, embed_dim)', shape='ellipse')
        
        if embed_dim != int(math.sqrt(embed_dim))**2:
            dot.node('linear_adjust', f'维度调整\n({embed_dim} → {int(math.sqrt(embed_dim))**2})', shape='box')
            dot.node('reshape', f'重塑为一维\n(batch*seq_len, 1, {int(math.sqrt(embed_dim))**2})', shape='box')
        else:
            dot.node('reshape', f'重塑为一维\n(batch*seq_len, 1, {embed_dim})', shape='box')
        
        dot.node('conv', '一维卷积\n(kernel_size=3, padding=1)', shape='box')
        dot.node('reshape_back', '重塑回原始形状\n(batch, seq_len, ...)', shape='box')
        dot.node('projection', f'输出投影\n(... → {embed_dim})', shape='box')
        dot.node('output_embed', '卷积嵌入输出\n(batch, seq_len, embed_dim)', shape='ellipse')
        
        # 添加边
        dot.edge('input_embed', 'reshape' if embed_dim == int(math.sqrt(embed_dim))**2 else 'linear_adjust')
        
        if embed_dim != int(math.sqrt(embed_dim))**2:
            dot.edge('linear_adjust', 'reshape')
        
        dot.edge('reshape', 'conv')
        dot.edge('conv', 'reshape_back')
        dot.edge('reshape_back', 'projection')
        dot.edge('projection', 'output_embed')
        
        # 保存图形
        dot.format = 'png'
        dot.render('conv_embedding_diagram', cleanup=True)
        print(f"卷积嵌入层结构图已保存为 'conv_embedding_diagram.png'")
    except Exception as e:
        print(f"无法创建卷积嵌入层结构图: {e}")

print("\n创建卷积嵌入层详细结构图...")
create_conv_embedding_diagram()