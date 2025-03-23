import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from ex2 import Net  # 导入模型
import matplotlib.pyplot as plt
import numpy as np

# 超参数
BATCH_SIZES = [16, 32, 48, 64, 80, 96, 112, 128]
LEARNING_RATES = [1e-4, 2e-4, 4e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3]
EPOCHS = 15
KEEP_PROB_RATE = 0.7

# 设备选择
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 下载 MNIST 数据集
path = './data/'
trainData = datasets.MNIST(path, train=True, transform=transform, download=True)
testData = datasets.MNIST(path, train=False, transform=transform)

# 记录结果
results = []

# 训练过程
for batch_size in BATCH_SIZES:
    for lr in LEARNING_RATES:
        # 构建数据加载器
        trainDataLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
        testDataLoader = DataLoader(testData, batch_size=batch_size)

        # 初始化模型
        net = Net(KEEP_PROB_RATE).to(device)

        # 损失函数 & 优化器
        lossF = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)

        # 记录最佳准确率
        best_accuracy = 0.0

        totalTrainLoss, totalTrainCorrect, totalTrainSamples = 0, 0, 0
        correct, totalLoss = 0, 0
        net.train()

        # 训练过程
        for epoch in range(1, EPOCHS + 1):
            net.train()
            totalTrainLoss, totalTrainCorrect, totalTrainSamples = 0, 0, 0
            processBar = tqdm(trainDataLoader, unit='step')

            for trainImgs, labels in processBar:
                trainImgs, labels = trainImgs.to(device), labels.to(device)

                # 前向传播
                outputs = net(trainImgs)
                loss = lossF(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算准确率
                predictions = torch.argmax(outputs, dim=1)
                batchCorrect = torch.sum(predictions == labels).item()

                totalTrainLoss += loss.item()
                totalTrainCorrect += batchCorrect
                totalTrainSamples += labels.shape[0]

                processBar.set_description(f"[{epoch}/{EPOCHS}] Loss: {loss.item():.4f}, Acc: {batchCorrect/labels.shape[0]:.4f}")

            epochTrainLoss = totalTrainLoss / len(trainDataLoader)
            epochTrainAccuracy = totalTrainCorrect / totalTrainSamples

            # 进入测试阶段
            correct, totalLoss = 0, 0
            net.eval()

            with torch.no_grad():
                for testImgs, labels in testDataLoader:
                    testImgs, labels = testImgs.to(device), labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)

                    predictions = torch.argmax(outputs, dim=1)
                    totalLoss += loss.item()
                    correct += torch.sum(predictions == labels).item()

            testAccuracy = correct / len(testData)
            testLoss = totalLoss / len(testDataLoader)

            # 保存最佳准确率
            if testAccuracy > best_accuracy:
                best_accuracy = testAccuracy

        # 记录结果
        results.append((batch_size, lr, best_accuracy))

# 将结果转为数组
results = np.array(results)

# 绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X 轴为 batch_size, Y 轴为 lr, Z 轴为 accuracy
x = results[:, 0]
y = results[:, 1]
z = results[:, 2]

ax.scatter(x, y, z, c=z, cmap='viridis')

ax.set_xlabel('Batch Size')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Accuracy')

# 显示图像
plt.show()
plt.savefig('./logs/img/train2.png')