import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from ex2 import Net  # 导入模型
from logger import setup_logger

# 设置日志记录器
logger = setup_logger(log_dir="logs", log_filename="training_cv.log")

# 超参数
BATCH_SIZE = 100
EPOCHS = 10
LEARNING_RATE = 1e-4
KEEP_PROB_RATE = 0.7
K_FOLDS = 5  # K折交叉验证的折数

# 设备选择
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 下载 MNIST 数据集
path = './data/'
dataset = datasets.MNIST(path, train=True, transform=transform, download=True)

# KFold 交叉验证
kf = KFold(n_splits=K_FOLDS, shuffle=True)

# 存储每一折的准确率
fold_accuracies = []

# 交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    logger.info(f"Fold {fold + 1}/{K_FOLDS}")
    
    # 创建训练集和验证集
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    # 初始化模型
    net = Net(KEEP_PROB_RATE).to(device)

    # 损失函数和优化器
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 训练过程
    for epoch in range(1, EPOCHS + 1):
        net.train()
        totalTrainLoss, totalTrainCorrect, totalTrainSamples = 0, 0, 0

        for trainImgs, labels in train_loader:
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

        epochTrainLoss = totalTrainLoss / len(train_loader)
        epochTrainAccuracy = totalTrainCorrect / totalTrainSamples
        logger.info(f"Epoch {epoch}: Train Loss: {epochTrainLoss:.4f}, Train Accuracy: {epochTrainAccuracy:.4f}")

    # 进入验证阶段
    net.eval()
    correct, totalLoss = 0, 0

    with torch.no_grad():
        for valImgs, labels in val_loader:
            valImgs, labels = valImgs.to(device), labels.to(device)
            outputs = net(valImgs)
            loss = lossF(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            totalLoss += loss.item()
            correct += torch.sum(predictions == labels).item()

    valAccuracy = correct / len(val_subset)
    valLoss = totalLoss / len(val_loader)
    logger.info(f"Fold {fold + 1} - Validation Loss: {valLoss:.4f}, Validation Accuracy: {valAccuracy:.4f}")
    
    # 记录每一折的验证准确率
    fold_accuracies.append(valAccuracy)

# 绘制交叉验证的准确率图
plt.figure(figsize=(10, 6))
plt.plot(range(1, K_FOLDS + 1), fold_accuracies, marker='o')
plt.title("Cross-Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("logs/img/cv_accuracy.png")
plt.show()

logger.info(f"Cross-validation results saved at logs/img/cv_accuracy.png")