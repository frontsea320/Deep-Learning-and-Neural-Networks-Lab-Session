import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_metrics(history, save_path=None):
    """
    绘制训练过程中的损失曲线和准确率曲线。
    
    参数：
    - history: dict，包含 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'。
    - save_path: str，可选，保存图片的路径。
    """
    epochs = range(1, len(history['Train Loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['Train Loss'], label="Train Loss", marker="o")
    plt.plot(epochs, history['Test Loss'], label="Test Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['Train Accuracy'], label="Train Accuracy", marker="o")
    plt.plot(epochs, history['Test Accuracy'], label="Test Accuracy", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Metrics saved to {save_path}")
    plt.show()


import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, data_loader, device, num_samples=8, save_path=None):
    """
    可视化模型的分类结果。
    
    参数：
    - model: 训练好的 PyTorch 模型。
    - data_loader: 测试数据的 DataLoader。
    - device: 设备（"cuda" 或 "cpu"）。
    - num_samples: 可视化的样本数量，默认为 8。
    - save_path: str，可选，保存图像的路径。
    """
    model.eval()
    images, labels = next(iter(data_loader))  # 获取一批数据
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    # 选取前 num_samples 个样本
    images, labels, predictions = images[:num_samples], labels[:num_samples], predictions[:num_samples]

    # 处理图像显示
    images = images.cpu().numpy()
    images = np.squeeze(images, axis=1)  # 去掉通道维度（灰度图）

    plt.figure(figsize=(10, 4))
    
    for i in range(num_samples):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(images[i], cmap="gray")
        
        # 在每张图片上添加标题
        plt.title(f"True: {labels[i].item()} | Pred: {predictions[i].item()}")
        
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f"Predictions visualization saved to {save_path}")
    plt.show()