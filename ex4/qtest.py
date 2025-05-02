import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from qmodel import QuaternionBasenjiModel
from qdata import load_dataset_from_excel, GeneDataset
from qtrain import QuaternionRegressionLoss
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 定义设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型和数据加载
def load_model_and_test(model_path, test_data_path):
    # 加载数据
    df = load_dataset_from_excel(test_data_path)
    test_dataset = GeneDataset(df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 假设所有样本的序列长度一致
    sample_seq, _ = test_dataset[0]
    seq_length = sample_seq.shape[1]

    # 构建模型
    model = QuaternionBasenjiModel(input_channels=4, seq_length=seq_length)
    model.to(device)  # 移动模型到适当设备（GPU 或 CPU）
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # 评估模型
    test_loss_all = 0
    all_preds = []
    all_labels = []

    # 使用 QuaternionRegressionLoss 作为损失函数
    loss_fn = QuaternionRegressionLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            # 前向传播
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss_all += loss.item() * inputs.size(0)

            # 收集预测结果和真实标签
            all_preds.extend(outputs.cpu().flatten().tolist())
            all_labels.extend(labels.cpu().flatten().tolist())

    avg_test_loss = test_loss_all / len(test_dataset)
    test_pcc, _ = pearsonr(all_preds, all_labels)

    # 输出结果
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test PCC: {test_pcc:.4f}")

    # 绘制预测 vs 真实标签的散点图
    plt.figure(figsize=(6,6))
    plt.scatter(all_preds, all_labels, s=0.5, alpha=0.7)
    plt.xlabel('Predicted TPM')
    plt.ylabel('True TPM')
    plt.title(f"Test Scatter Plot\nPCC: {test_pcc:.4f}")
    plt.savefig('test_plot.png', dpi=300)
    plt.show()

# 主函数
def main():
    model_path = 'best_qmodel.pth'  # 替换为你的模型路径
    test_data_path = './dataset.xlsx'  # 替换为你的测试数据文件路径
    load_model_and_test(model_path, test_data_path)

if __name__ == '__main__':
    main()
