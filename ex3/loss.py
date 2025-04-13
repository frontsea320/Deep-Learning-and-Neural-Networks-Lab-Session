import numpy as np
import matplotlib.pyplot as plt

# 加载保存的loss记录
loss_history = np.load('loss_history.npy')

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='train_Loss', color='lightblue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss_Avg')
plt.title('Loss decline curve during training')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png')  # 保存为图片
plt.show()
