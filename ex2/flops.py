import torch
from torchsummary import summary
from thop import profile
from improve_ex2 import improve_Net  # 导入改进后的模型
from ex2 import Net  # 导入原始的 Net 模型

# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
improve_net = improve_Net(keep_prob=0.7).to(device)
net = Net(keep_prob=0.7).to(device)

# 定义一个示例输入张量
input_tensor = torch.randn(1, 1, 28, 28).to(device)  # 假设输入是一个28x28的单通道图像

# 打印模型参数量
print("\n=== ImprovedNet Parameters ===")
summary(improve_net, input_size=(1, 28, 28))

print("\n=== Net Parameters ===")
summary(net, input_size=(1, 28, 28))

# 计算浮点运算次数（FLOPs）和参数量
def print_flops(model, input_tensor):
    # 计算模型的参数量和FLOPs
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

# 打印 ImprovedNet 的 FLOPs 和参数量
print("\n=== ImprovedNet FLOPs and Params ===")
print_flops(improve_net, input_tensor)

# 打印 Net 的 FLOPs 和参数量
print("\n=== Net FLOPs and Params ===")
print_flops(net, input_tensor)