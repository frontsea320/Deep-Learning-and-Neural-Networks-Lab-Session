# test.py
import torch
from model import TextConverter, myRNN
import numpy as np
import re

# Load trained model
use_gpu = True
text_path = './poetry.txt'
convert = TextConverter(text_path, max_vocab=10000)
model = myRNN(convert.vocab_size, 512, 512, 2, 0.5)
model.load_state_dict(torch.load('model.pth'))
if use_gpu:
    model = model.cuda()

# Sampling function
def pick_top_n(preds, top_n=3):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c

# Input
begin = input('## 请输入第一个字 please input the first character: ')
text_len = 28

# Evaluate mode
model.eval()
samples = [convert.word_to_int(c) for c in begin]
input_txt = torch.LongTensor(samples)[None]
if use_gpu:
    input_txt = input_txt.cuda()
input_txt = torch.autograd.Variable(input_txt)

# Ensure hidden state is on the same device as input
hs = torch.zeros(model.num_layers, input_txt.size(0), model.hidden_size).to(input_txt.device)  # Move hidden state to the same device as input

_, init_state = model(input_txt, hs)
result = samples
model_input = input_txt[:, -1][:, None]

i = 0
while begin != 0:
    out, init_state = model(model_input, init_state)
    pred = pick_top_n(out.data)

    model_input = torch.autograd.Variable(torch.LongTensor(pred))[None]
    model_input = model_input.cuda()
    if pred[0] != 0:
        result.append(pred[0])
    i += 1
    if i > text_len - 2:
        break

# Output
text = convert.arr_to_text(result)
text = re.findall(r'.{7}', text)
text = '，'.join(text) + '。'

print(f'输出：\n{text}')