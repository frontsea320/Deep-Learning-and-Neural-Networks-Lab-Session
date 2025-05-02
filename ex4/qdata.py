# data_utils.py
import torch
from torch.utils.data import Dataset
import pandas as pd

def load_dataset_from_excel(file_path):
    """
    从 Excel 文件中读取数据，合并所有 sheet。
    文件中应包含以下列:
      - Geneid
      - sequence (DNA 序列字符串)
      - TPM (基因表达量，回归目标)
      - dataset (可选：若存在 'train' 和 'test' 标记则用作划分)
    返回合并后的 DataFrame。
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    return df

def split_dataset(df, split_col='dataset'):
    """
    根据 DataFrame 中的 split_col 划分数据。
    若该列存在且取值为 'train' 或 'test'，则按照该列划分；否则按 80%/20% 随机划分。
    返回: train_df, test_df
    """
    if split_col in df.columns and set(df[split_col].unique()).issuperset({'train', 'test'}):
        train_df = df[df[split_col]=='train'].reset_index(drop=True)
        test_df = df[df[split_col]=='test'].reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size].reset_index(drop=True)
        test_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df

def encode_sequence(seq):
    """
    将 DNA 序列转换为 one-hot 编码张量，输出 shape 为 (4, L)
    编码规则:
      A -> [1, 0, 0, 0]
      T -> [0, 1, 0, 0]
      C -> [0, 0, 1, 0]
      G -> [0, 0, 0, 1]
    """
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1]}
    seq = seq.upper()
    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in seq]
    tensor = torch.tensor(encoded, dtype=torch.float32).transpose(0, 1)
    return tensor

class GeneDataset(Dataset):
    def __init__(self, df):
        """
        df: DataFrame，至少包含 'sequence' 和 'TPM' 两列。
        TPM 被作为回归目标。
        """
        self.sequences = df['sequence'].tolist()
        self.labels = df['TPM'].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_tensor = encode_sequence(self.sequences[idx])
        # 确保标签为 float32
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return seq_tensor, label