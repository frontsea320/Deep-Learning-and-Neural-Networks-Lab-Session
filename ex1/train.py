import argparse
from ex1 import MyModel, compute_accuracy, mnist_dataset, train_one_step, test
from plot_metrics import plot_metrics
from save_weights import save_weights
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser(description="Train model with optional pretrained weights")
parser.add_argument('--weights_dir', type=str, default=None, help='Path to pretrained weights')
args = parser.parse_args()

# 主训练流程
model = MyModel(pretrained_dir=args.weights_dir)
train_data, test_data = mnist_dataset()
train_label = np.zeros(shape=[train_data[0].shape[0], 10])
test_label = np.zeros(shape=[test_data[0].shape[0], 10])
train_label[np.arange(train_data[0].shape[0]), train_data[1]] = 1
test_label[np.arange(test_data[0].shape[0]), test_data[1]] = 1

initial_lr = 5e-5  # 初始学习率
loss_list, accuracy_list = [], []
for epoch in range(150):
    # 计算当前 epoch 的学习率（指数衰减）
    learning_rate = initial_lr #* (0.99 ** epoch)

    loss, accuracy = train_one_step(model, train_data[0], train_label, learning_rate=learning_rate)

    loss_list.append(loss)
    accuracy_list.append(accuracy)
    print(f'Epoch {epoch}: Loss {loss:.4f}; Accuracy {accuracy:.4f}; LR {learning_rate:.6f}')

plot_metrics(loss_list, accuracy_list, save_dir="homework/ex1/loss")

loss, accuracy = test(model, test_data[0], test_label)
print(f'Test Loss {loss}; Accuracy {accuracy}')

save_weights(model, save_dir="homework/ex1/weight")
print('权重文件已保存到 homework/ex1/weight')
