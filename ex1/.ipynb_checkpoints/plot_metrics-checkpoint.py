import matplotlib.pyplot as plt
import os

def plot_metrics(loss_list, accuracy_list, save_dir="plots"):
    epochs = range(1, len(loss_list) + 1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 5))

    # Loss变化图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_list, 'b-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率变化图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_list, 'r-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.show()
