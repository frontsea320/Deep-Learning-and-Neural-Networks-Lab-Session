import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

# 导入模型定义
from model import BasenjiModel, MyDataset

def train_model(train_data, y_train, valid_data, y_valid, test_data, y_test, 
               batch_size=32, epochs=10, lr=1e-3, device=None):
    """
    训练并评估基因表达预测模型
    
    参数:
    train_data: 训练集DNA序列的one-hot编码
    y_train: 训练集基因表达量
    valid_data: 验证集DNA序列的one-hot编码
    y_valid: 验证集基因表达量
    test_data: 测试集DNA序列的one-hot编码
    y_test: 测试集基因表达量
    batch_size: 批量大小
    epochs: 训练轮数
    lr: 学习率
    device: 训练设备(cuda或cpu)
    
    返回:
    trained_model: 训练好的模型
    history: 训练历史记录
    """
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 数据加载器
    train_dataset = MyDataset(train_data, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = MyDataset(valid_data, y_valid)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    test_dataset = MyDataset(test_data, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    # 初始化模型
    net = BasenjiModel()
    print(f"模型参数总数: {sum(p.numel() for p in net.parameters())}")
    net = net.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # 存储训练历史
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_pcc': []
    }
    
    best_valid_pcc = -1
    best_model_path = "best_model.pth"
    
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        # 训练阶段
        net.train(True)
        process_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        epoch_loss_all = 0
        
        for inputs, labels in process_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = net(inputs)
            loss = loss_fn(outputs.reshape(labels.shape), labels)
            epoch_loss_all += loss.item() * len(labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            process_bar.set_description(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
        
        train_loss = epoch_loss_all / len(train_dataset)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        net.train(False)
        valid_loss_all = 0
        y_valid_pred_all = []
        y_valid_true_all = []
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = net(inputs)
                loss = loss_fn(outputs.reshape(labels.shape), labels)
                valid_loss_all += loss.item() * len(labels)
                
                y_valid_pred_all.extend(outputs.flatten().cpu().tolist())
                y_valid_true_all.extend(labels.cpu().tolist())
        
        valid_loss = valid_loss_all / len(valid_dataset)
        valid_pcc, _ = pearsonr(y_valid_pred_all, y_valid_true_all)
        
        history['valid_loss'].append(valid_loss)
        history['valid_pcc'].append(valid_pcc)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid PCC: {valid_pcc:.4f}")
        
        # 保存最佳模型
        if valid_pcc > best_valid_pcc:
            best_valid_pcc = valid_pcc
            torch.save(net.state_dict(), best_model_path)
            print(f"保存新的最佳模型，验证集PCC: {valid_pcc:.4f}")
    
    # 加载最佳模型进行测试
    print(f"加载最佳模型进行测试评估...")
    net.load_state_dict(torch.load(best_model_path))
    net.eval()
    
    # 测试阶段
    test_loss_all = 0
    y_test_pred_all = []
    y_test_true_all = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(inputs)
            loss = loss_fn(outputs.reshape(labels.shape), labels)
            test_loss_all += loss.item() * len(labels)
            
            y_test_pred_all.extend(outputs.flatten().cpu().tolist())
            y_test_true_all.extend(labels.cpu().tolist())
    
    test_loss = test_loss_all / len(test_dataset)
    test_pcc, _ = pearsonr(y_test_pred_all, y_test_true_all)
    
    print(f"测试集结果 - Loss: {test_loss:.4f}, PCC: {test_pcc:.4f}")
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_pred_all, y_test_true_all, s=1, alpha=0.5)
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title(f'测试集散点图 (PCC = {test_pcc:.4f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加对角线
    max_val = max(max(y_test_pred_all), max(y_test_true_all))
    min_val = min(min(y_test_pred_all), min(y_test_true_all))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.tight_layout()
    plt.savefig('test_scatter.png')
    plt.close()
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['valid_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['valid_pcc'], label='验证PCC')
    plt.axhline(y=test_pcc, color='r', linestyle='--', label=f'测试PCC={test_pcc:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('PCC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # 保存训练历史和测试结果
    results = {
        'history': history,
        'test_pcc': test_pcc,
        'test_loss': test_loss,
        'y_test_pred': y_test_pred_all,
        'y_test_true': y_test_true_all
    }
    
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return net, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练玉米基因表达量预测模型')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--data_path', type=str, default='processed_data.pkl', help='处理后的数据路径')
    args = parser.parse_args()
    
    print("加载数据...")
    try:
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        
        train_data = data['train_data']
        y_train = data['y_train']
        valid_data = data['valid_data']
        y_valid = data['y_valid']
        test_data = data['test_data']
        y_test = data['y_test']
        
        print(f"数据加载成功 - 训练集: {len(train_data)}, 验证集: {len(valid_data)}, 测试集: {len(test_data)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请先运行数据预处理脚本生成processed_data.pkl文件")
        sys.exit(1)
    
    # 训练模型
    model, history = train_model(
        train_data, y_train, 
        valid_data, y_valid, 
        test_data, y_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    print("训练和评估完成！")
    print(f"结果已保存到 'training_results.pkl'")
    print(f"散点图已保存到 'test_scatter.png'")
    print(f"训练历史图已保存到 'training_history.png'")