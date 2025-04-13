import torch
import argparse
import numpy as np
from improve_model import ConvRNN

def sample(model, text_converter, start_text, length=100, temperature=0.8, device='cpu'):
    """
    使用训练好的模型生成文本
    Args:
        model: 训练好的模型
        text_converter: 文本转换器
        start_text: 生成文本的起始字符
        length: 要生成的文本长度
        temperature: 温度参数，控制随机性。较低的温度会使生成更确定性，较高的温度增加多样性
        device: 使用的设备
    """
    model.eval()  # 切换到评估模式
    
    # 初始化隐藏状态
    hidden = None
    
    # 将起始文本转换为索引
    start_arr = text_converter.text_to_arr(start_text)
    x = torch.tensor([start_arr], dtype=torch.long).to(device)
    
    # 用于保存生成的文本
    generated_text = start_text
    
    # 逐个字符生成
    for i in range(length):
        # 前向传播
        outputs, hidden = model(x, hidden)
        outputs = outputs[-1].view(1, -1)  # 取最后一个时间步的输出
        
        # 应用温度参数
        if temperature != 1.0:
            outputs = outputs / temperature
        
        # 将输出转换为概率分布
        probs = torch.softmax(outputs, dim=1)
        
        # 根据概率分布采样下一个字符
        next_char_idx = torch.multinomial(probs, 1).item()
        
        # 将字符添加到生成的文本中
        next_char = text_converter.int_to_word(next_char_idx)
        generated_text += next_char
        
        # 更新输入
        x = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--converter_path', type=str, default='./weights/improve_checkpoints/text_converter.pth', help='文本转换器路径')
    parser.add_argument('--start_text', type=str, default='春', help='生成文本的起始字符')
    parser.add_argument('--length', type=int, default=200, help='要生成的文本长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--num_samples', type=int, default=5, help='生成样本的数量')
    args = parser.parse_args()
    
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载文本转换器
    text_converter = torch.load(args.converter_path)
    print(f"已加载文本转换器，词汇表大小: {text_converter.vocab_size}")
    
    # 加载模型参数
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 重建模型
    model = ConvRNN(
        num_classes=checkpoint['vocab_size'],
        embed_dim=checkpoint['embedding_dim'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"已加载模型: {args.model_path}")
    
    # 生成多个样本
    for i in range(args.num_samples):
        generated_text = sample(
            model=model,
            text_converter=text_converter,
            start_text=args.start_text,
            length=args.length,
            temperature=args.temperature,
            device=device
        )
        
        print(f"\n样本 {i+1}:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)

if __name__ == '__main__':
    main()