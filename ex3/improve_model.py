import torch
from torch import nn
from torch.autograd import Variable
import math

class TextConverter(object):
    def __init__(self, text_path, max_vocab=5000):
        """
        建立一个字符索引转换器
        Args:
            text_path: 文本位置
            max_vocab: 最大的单词数量
        """
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace('\n', ' ').replace('\r', ' ') \
            .replace('，', ' ').replace('。', ' ')
        # 去掉重复的字符
        vocab = set(text)
        # 如果单词总数超过最大值，去掉频率最低的
        vocab_count = {}
        for word in vocab:
            vocab_count[word] = 0
        for word in text:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return ','
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return arr

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

class ConvolutionalWordEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 确保嵌入维度是完全平方数
        self.sqrt_dim = int(math.sqrt(embed_dim))
        self.squared_dim = self.sqrt_dim ** 2
        
        # 如果嵌入维度不是完全平方数，我们需要调整
        if self.squared_dim != embed_dim:
            self.linear_adjust = nn.Linear(embed_dim, self.squared_dim)
        
        # 卷积层：采用小卷积核并保持输出维度接近原始维度
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,  # 使用3x1的卷积核
            stride=1,
            padding=1  # 使用padding保持尺寸
        )
        
        # 用于将卷积输出映射回原始嵌入维度
        self.output_projection = nn.Linear(self.squared_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # 保持原始形状，但在通道维度上应用卷积
        if self.squared_dim != embed_dim:
            # 调整嵌入维度
            x = self.linear_adjust(x)
        
        # 重塑为 (batch_size * seq_len, 1, squared_dim) 以便应用1D卷积
        x_reshaped = x.view(batch_size * seq_len, 1, self.squared_dim)
        
        # 应用卷积
        conv_out = self.conv(x_reshaped)
        
        # 重塑回原始形状
        conv_out = conv_out.view(batch_size, seq_len, self.squared_dim)
        
        # 投影回原始嵌入维度
        output = self.output_projection(conv_out)
        
        return output

class ConvRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        
        # 初始词嵌入层
        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        
        # 卷积层处理词嵌入
        self.conv_embed = ConvolutionalWordEmbedding(embed_dim)
        
        # RNN层
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, dropout=dropout)
        
        # 投影层（输出层）
        self.project = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hs=None):
        batch = x.shape[0]
        seq_len = x.shape[1]
        
        if hs is None:
            hs = Variable(torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device))
        
        # 词嵌入
        word_embed = self.word_to_vec(x)  # (batch, seq_len, embed_dim)
        
        # 应用卷积变换，保持序列长度不变
        conv_embed = self.conv_embed(word_embed)  # (batch, seq_len, embed_dim)
        
        # 调整形状以适应RNN输入 (seq_len, batch, embed_dim)
        rnn_input = conv_embed.permute(1, 0, 2)
        
        # RNN处理
        out, h0 = self.rnn(rnn_input, hs)  # (seq_len, batch, hidden)
        
        # 输出处理
        le, mb, hd = out.shape
        out = out.reshape(le * mb, hd)
        out = self.project(out)
        out = out.reshape(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, seq_len, num_classes)
        
        return out.reshape(-1, out.shape[2]), h0