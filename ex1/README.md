# HZAU_2025-深度学习与神经网络课程实验一：MNIST 手写数字识别神经网络模型

## 项目简介
本项目基于原始实现MNIST数据集手写数字识别任务的代码，做了更利好新手使用的修改。

---

## 🚀 项目功能与特点

- **基础神经网络结构**：
  - 输入层 → 隐藏层 (100维) → 隐藏层 (66维) → 输出层（10类）。
  - 激活函数采用ReLU和Softmax。
  - 使用交叉熵损失函数。

- **残差连接**：
  - 在隐藏层之间新增了一个残差连接（Residual Connection），以改善梯度传播，提高模型性能。

- **衰减学习率**：
  - ~~引入指数型学习率衰减策略，每个 epoch 学习率自动降低，优化训练效果。~~
  - 2025.3.8更新：我们发现采用固定学习率进行训练可以达到更好的效果，所以代码中使用固定学习率，但保留了使用衰减学习率的功能，可以按需使用

- **训练与测试分离**：
  - 将模型定义、训练过程和测试过程分开，保留模型定义的训练功能（更利好使用）。

- **权重文件存储与加载**：
  - 提供简单的权重保存与加载机制，方便复用已训练的权重。
  
- **预训练权重使用**：
  - 提供初始预训练权重，避免重复训练。

- **结果可视化**：
  - 提供 loss 与 accuracy 曲线图，直观观察训练过程。
  - 测试阶段随机抽取若干张测试集图片并输出模型预测结果，便于直观分析。

---

## 📂 项目结构

```
/project_root
│── loss/                             # loss曲线图像的保存路径
│── pretrained_weights/               # 预训练权重的保存路径
│── test_img/                         # 测试图片保存路径
│── weight/                           # 训练权重存储路径
│── ex1.py                            # 模型定义
│── load_and_visualize.py             # 权重加载与预测可视化脚本
│── plot_metrics.py                   # 训练结果可视化
│── README.md                         # 自述文件
│── requirements.txt                  # 项目运行所需的依赖文件
│── save_weights.py                   # 权重保存脚本
│── test5.py                          # 使用训练权重测试模型
│── train.py                          # 模型训练脚本
```

---

## ⚙️ 使用说明

本实验采用单卡Nvidia-2080ti进行训练

### ① 环境准备
```bash
pip install -r requirements.txt
```

### ② 训练模型

```bash
python train.py
```
- **使用预训练权重继续训练**：
```bash
python train.py --weights_dir weights
```

### ② 测试并可视化结果
```bash
python test.py --weights_dir weights --save_path visualization --num_samples 5
```

---
## 📌 输出结果说明

- 模型训练时，loss和accuracy曲线将自动保存至`loss/`。
- 训练好的权重将保存在`weight/`目录中。
- 测试时预测结果图片保存在指定路径（默认：`test_img/`）。

---

## 🎯 未来改进方向
- 我有一个绝妙的想法，可惜 cuda out of memory

---

🎉 **从何时惋惜蝴蝶困于那桃源**

