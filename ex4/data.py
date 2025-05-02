import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# One-hot编码函数
def one_hot_encode_along_channel_axis(sequence):
    """将DNA序列（ACGT）转换为one-hot编码"""
    sequence = sequence.upper()
    seq_len = len(sequence)
    one_hot = np.zeros((4, seq_len), dtype=np.int8)
    
    for i in range(seq_len):
        if sequence[i] == 'A':
            one_hot[0, i] = 1
        elif sequence[i] == 'C':
            one_hot[1, i] = 1
        elif sequence[i] == 'G':
            one_hot[2, i] = 1
        elif sequence[i] == 'T':
            one_hot[3, i] = 1
    
    return one_hot

# 读取数据集
print("开始读取数据集...")
df = pd.read_excel('./dataset.xlsx')
print('基因总数:', df.shape[0])

# 检查数据集中是否已有划分
if 'dataset' not in df.columns:
    print("数据集中没有划分信息，正在进行划分...")
    # 使用sklearn进行数据集划分
    train_indices, test_indices = train_test_split(
        range(len(df)), 
        test_size=0.2, 
        random_state=2023
    )
    train_indices, valid_indices = train_test_split(
        train_indices, 
        test_size=0.1, 
        random_state=2023
    )
    
    # 添加划分标签
    df['dataset'] = None
    df.loc[train_indices, 'dataset'] = 'train'
    df.loc[valid_indices, 'dataset'] = 'valid'
    df.loc[test_indices, 'dataset'] = 'test'
    
    # 保存带划分信息的数据集
    df.to_excel('dataset_split.xlsx', index=False)
    print("数据集划分已保存到dataset_split.xlsx")
else:
    print("使用现有的数据集划分信息")

# 处理训练集
print("处理训练集...")
df_train = df[df['dataset'] == 'train']
y_train = np.log2(df_train['TPM'].values + 1)
train_data = np.array([one_hot_encode_along_channel_axis(i.strip()) for i in df_train['sequence'].values])

# 处理验证集
print("处理验证集...")
df_valid = df[df['dataset'] == 'valid']
y_valid = np.log2(df_valid['TPM'].values + 1)
valid_data = np.array([one_hot_encode_along_channel_axis(i.strip()) for i in df_valid['sequence'].values])

# 处理测试集
print("处理测试集...")
df_test = df[df['dataset'] == 'test']
y_test = np.log2(df_test['TPM'].values + 1)
test_data = np.array([one_hot_encode_along_channel_axis(i.strip()) for i in df_test['sequence'].values])

# 打印数据集大小
print(f"训练集大小: {len(train_data)} 样本")
print(f"验证集大小: {len(valid_data)} 样本")
print(f"测试集大小: {len(test_data)} 样本")

# 保存处理后的数据
print("保存处理后的数据...")
with open('processed_data.pkl', 'wb') as f:
    pickle.dump({
        'train_data': train_data,
        'y_train': y_train,
        'valid_data': valid_data,
        'y_valid': y_valid,
        'test_data': test_data,
        'y_test': y_test
    }, f)

print("数据预处理完成，已保存到processed_data.pkl")