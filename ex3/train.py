# train.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import TextConverter, myRNN

# Hyperparameters
learning_rate = 1e-4
max_epoch = 20
batch_size = 128
use_gpu = True
text_path = './poetry.txt'

# Data Preprocessing
convert = TextConverter(text_path, max_vocab=10000)
poetry_corpus = open(text_path, 'r').read()
poetry_corpus = poetry_corpus.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')

# Convert poetry corpus into integer arrays
n_step = 20
num_seq = len(poetry_corpus) // n_step
text = poetry_corpus[:num_seq * n_step]
arr = np.array(convert.text_to_arr(text)).reshape((num_seq, -1))
arr = torch.from_numpy(arr)

# Dataset Class
class TextDataset(object):
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        x = self.arr[item, :]
        y = torch.zeros(x.shape)
        y[:-1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.arr.shape[0]

train_set = TextDataset(arr)
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Model and Optimizer
model = myRNN(convert.vocab_size, 512, 512, 2, 0.5)
if use_gpu:
    model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for e in range(max_epoch):
    train_loss = 0
    for data in train_data:
        x, y = data
        y = y.long()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)

        # Ensure hidden state is on the same device as input
        hs = torch.zeros(model.num_layers, x.size(0), model.hidden_size).to(x.device)  # Move hidden state to the same device as input

        # Forward
        score, _ = model(x, hs)
        loss = criterion(score, y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch: {e+1}, Perplexity: {np.exp(train_loss / len(train_data)):.3f}, Loss: {train_loss / batch_size:.3f}')

# Save model weights at the end of each epoch (or after final epoch)
torch.save(model.state_dict(), 'weights/model.pth')  # Save model weights
print(f'Model weights saved to weights/model.pth after epoch {e+1}')