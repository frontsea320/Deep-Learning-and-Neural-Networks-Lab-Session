from load_and_visualize import load_weights, visualize_predictions

# 加载权重
load_weights(model, weights_dir="你的权重文件夹路径")

# 随机测试并可视化五张图片
visualize_predictions(model, num_samples=5, save_path="你的保存路径")