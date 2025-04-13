import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import time
import argparse

# 导入修复后的模型
# 请确保将修复后的模型保存为 improve_model.py
from improve_model import TextConverter, ConvRNN

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--text_path', type=str, default='./poetry.txt', help='文本路径')
parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
parser.add_argument('--embedding_dim', type=int, default=256, help='词嵌入维度')
parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层大小')
parser.add_argument('--num_layers', type=int, default=2, help='RNN层数')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout概率')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
parser.add_argument('--seq_length', type=int, default=48, help='序列长度')
parser.add_argument('--save_dir', type=str, default='weights/improve_checkpoints', help='模型保存路径')
parser.add_argument('--save_every', type=int, default=1, help='每多少轮保存一次模型')
args = parser.parse_args()

# 创建保存模型的目录
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# 文本数据集类
class TextDataset(Dataset):
    def __init__(self, text_path, seq_length, text_converter):
        self.seq_length = seq_length
        self.text_converter = text_converter
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 将文本转换为整数序列
        self.text_arr = text_converter.text_to_arr(text)
        self.text_arr = torch.tensor(self.text_arr)
        
        # 计算有多少个样本
        self.num_samples = len(self.text_arr) - self.seq_length
        
        print(f"文本总长度: {len(self.text_arr)}, 样本数量: {self.num_samples}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 输入序列
        x = self.text_arr[idx:idx+self.seq_length]
        # 目标序列（向右移动一位）
        y = self.text_arr[idx+1:idx+self.seq_length+1]
        return x, y

def train():
    # 初始化文本转换器
    text_converter = TextConverter(args.text_path)
    vocab_size = text_converter.vocab_size
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集和数据加载器
    dataset = TextDataset(args.text_path, args.seq_length, text_converter)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 创建模型
    model = ConvRNN(
        num_classes=vocab_size,
        embed_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 开始训练
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            # 将数据移动到设备
            inputs = inputs.to(device)
            targets = targets.view(-1).to(device)
            
            # 前向传播
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印训练信息
            if (i+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, 用时: {time.time() - start_time:.2f}秒')
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}')
        
        # 保存模型
        if (epoch+1) % args.save_every == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embedding_dim': args.embedding_dim,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }
            torch.save(checkpoint, f'{args.save_dir}/model_epoch_{epoch+1}.pth')
            print(f'模型已保存: {args.save_dir}/model_epoch_{epoch+1}.pth')
    
    # 训练结束，保存最终模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': args.embedding_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    torch.save(checkpoint, f'{args.save_dir}/model_final.pth')
    print(f'最终模型已保存: {args.save_dir}/model_final.pth')
    
    # 同时保存文本转换器，以便测试时使用
    torch.save(text_converter, f'{args.save_dir}/text_converter.pth')
    print(f'文本转换器已保存: {args.save_dir}/text_converter.pth')
    
    print(f'总训练时间: {time.time() - start_time:.2f}秒')

if __name__ == '__main__':
    train()