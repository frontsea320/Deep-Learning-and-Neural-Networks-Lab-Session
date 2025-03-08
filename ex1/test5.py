import argparse
import numpy as np
from ex1 import MyModel, mnist_dataset, compute_accuracy
from load_and_visualize import load_weights, visualize_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='仅测试模型，不进行训练')
    parser.add_argument('--weights_dir', type=str, default="homework/ex1/weight", help='权重文件夹路径')
    parser.add_argument('--save_path', type=str, default="homework/ex1/test_img", help='预测图片保存路径')
    parser.add_argument('--num_samples', type=int, default=5, help='随机测试图片数量')

    args = parser.parse_args()

    # 加载模型和权重
    model = MyModel()
    load_weights(model, weights_dir=args.weights_dir)

    # 测试数据
    _, test_data = mnist_dataset()
    test_label = np.zeros(shape=[test_data[0].shape[0], 10])
    test_label[np.arange(test_data[0].shape[0]), test_data[1]] = 1

    # 测试
    model.forward(test_data[0], test_label)
    loss = model.loss
    accuracy = compute_accuracy(model.h3_soft, test_label)
    print(f'Test Loss: {loss}, Accuracy: {accuracy}')

    # 随机测试并可视化预测图片
    visualize_predictions(model, num_samples=args.num_samples, save_path=args.save_path)