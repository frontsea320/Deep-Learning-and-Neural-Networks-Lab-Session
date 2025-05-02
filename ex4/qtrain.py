import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from qmodel import QuaternionBasenjiModel
from qdata import load_dataset_from_excel, GeneDataset
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import logging

# 设置日志记录
def setup_logger():
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('training_log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

class QuaternionRegressionLoss(nn.Module):
    def __init__(self, alpha=4.0, beta=0):
        super(QuaternionRegressionLoss, self).__init__()
        self.alpha = alpha  # 权重，用于控制模损失的影响
        self.beta = beta    # 权重，用于控制余弦相似度损失的影响

    def forward(self, pred, target):
        # 计算四元数模差
        pred_norm = torch.sqrt(torch.sum(pred ** 2, dim=1))
        target_norm = torch.sqrt(torch.sum(target ** 2, dim=1))
        norm_loss = torch.mean((pred_norm - target_norm) ** 2)

        # 计算余弦相似度
        dot_product = torch.sum(pred * target, dim=1)
        pred_norm = torch.sqrt(torch.sum(pred ** 2, dim=1))
        target_norm = torch.sqrt(torch.sum(target ** 2, dim=1))
        cosine_similarity = dot_product / (pred_norm * target_norm + 1e-8)
        cosine_loss = torch.mean(1 - cosine_similarity)

        # 综合损失：模损失和余弦损失的加权和
        total_loss = self.alpha * norm_loss + self.beta * cosine_loss
        return total_loss

def main():
    # 初始化日志记录
    logger = setup_logger()

    # Excel文件路径，请根据实际路径调整
    excel_file = './dataset.xlsx'
    df = load_dataset_from_excel(excel_file)
    criterion = QuaternionRegressionLoss()
    
    # 如果数据中存在 "dataset" 列，并且包含 "train", "valid", "test"
    if 'dataset' in df.columns and set(df['dataset'].unique()).issuperset({'train', 'valid', 'test'}):
        train_df = df[df['dataset'] == 'train']
        valid_df = df[df['dataset'] == 'valid']
        test_df  = df[df['dataset'] == 'test']
    else:
        # 若数据中没有完整划分，则采用 split_dataset 得到 train 和 test，
        # 这里仅做示例，验证集可与 test 集相同（或自行拆分）
        print("未检测到完整的数据划分，采用默认划分方式。")
        train_df, test_df = split_dataset(df, split_col='dataset')
        valid_df = test_df  # 此时验证集与测试集相同
    
    print(f"训练样本数量: {len(train_df)}, 验证样本数量: {len(valid_df)}, 测试样本数量: {len(test_df)}")
    
    # 构造数据集
    train_dataset = GeneDataset(train_df)
    valid_dataset = GeneDataset(valid_df)
    test_dataset  = GeneDataset(test_df)
    
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 假设所有样本的序列长度一致，取第一个样本的长度（one-hot编码后的 shape 为 (4, L)）
    sample_seq, _ = train_dataset[0]
    seq_length = sample_seq.shape[1]
    print("序列长度：", seq_length)
    
    # 构建模型；输入通道数为4（one-hot编码），序列长度传入模型
    net = QuaternionBasenjiModel(input_channels=4, seq_length=seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # 损失函数和优化器
    loss_fn = criterion
    optimizer = optim.Adam(net.parameters(), lr=5e-3)
    EPOCHS = 50
    
    # 初始化记录变量
    history = {'Train Loss': [], 'Valid Loss': [], 'Valid PCC': []}
    best_valid_pcc = -1e9
    best_model_path = "best_qmodel.pth"
    
    for epoch in range(1, EPOCHS + 1):
        # ---------------------------
        # 训练阶段
        net.train()
        epoch_loss_all = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="step")
        for inputs, labels in pbar:
            inputs = inputs.to(device)            # shape: (batch, 4, seq_length)
            labels = labels.to(device).unsqueeze(1) # shape: (batch, 1)
            
            optimizer.zero_grad()
            outputs = net(inputs)                 # 输出 shape: (batch, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss_all += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss = epoch_loss_all / len(train_dataset)
        history['Train Loss'].append(train_loss)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        
        # 记录训练日志
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        
        # ---------------------------
        # 验证阶段
        net.eval()
        valid_loss_all = 0
        y_valid_pred_all = []
        y_valid_true_all = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                valid_loss_all += loss.item() * inputs.size(0)
                
                y_valid_pred_all.extend(outputs.cpu().flatten().tolist())
                y_valid_true_all.extend(labels.cpu().flatten().tolist())
        
        valid_loss = valid_loss_all / len(valid_dataset)
        valid_pcc, _ = pearsonr(y_valid_pred_all, y_valid_true_all)
        history['Valid Loss'].append(valid_loss)
        history['Valid PCC'].append(valid_pcc)
        print(f"Epoch {epoch}: Valid Loss: {valid_loss:.4f}, Valid PCC: {valid_pcc:.4f}")
        
        # 记录验证日志
        logger.info(f"Epoch {epoch}: Valid Loss: {valid_loss:.4f}, Valid PCC: {valid_pcc:.4f}")
        
        # 保存最佳模型（以验证集的 PCC 为评判标准）
        if valid_pcc > best_valid_pcc:
            best_valid_pcc = valid_pcc
            torch.save(net.state_dict(), best_model_path)
            print(f"保存新的最佳模型，验证集PCC: {valid_pcc:.4f}")
    
    print("训练完成，最佳模型已保存。")
    
    # --------------------------
    # 测试阶段：加载最佳模型，评估测试集表现
    print("加载最佳模型进行测试评估...")
    net.load_state_dict(torch.load(best_model_path))
    net.eval()
    test_loss_all = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            test_loss_all += loss.item() * inputs.size(0)
            
            all_preds.extend(outputs.cpu().flatten().tolist())
            all_labels.extend(labels.cpu().flatten().tolist())
    
    avg_test_loss = test_loss_all / len(test_dataset)
    test_pcc, _ = pearsonr(all_preds, all_labels)
    print(f"Test Loss: {avg_test_loss:.4f}, Test PCC: {test_pcc:.4f}")
    
    # 记录测试日志
    logger.info(f"Test Loss: {avg_test_loss:.4f}, Test PCC: {test_pcc:.4f}")
    
    # 绘制散点图：横轴为预测值，纵轴为真实值
    plt.figure(figsize=(6,6))
    plt.scatter(all_preds, all_labels, s=0.5, alpha=0.7)
    plt.xlabel('Predicted TPM')
    plt.ylabel('True TPM')
    plt.title(f"Test Scatter Plot\nPCC: {test_pcc:.4f}")
    plt.savefig('ori_plot.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
