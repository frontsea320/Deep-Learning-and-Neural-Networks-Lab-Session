import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from ex2 import Net  # 导入模型
from visualization import plot_metrics
from logger import setup_logger

# 设置日志记录器
logger = setup_logger(log_dir="logs", log_filename="training_base.log")

# 记录训练和测试的历史数据
history = {
    "Train Loss": [],
    "Train Accuracy": [],
    "Test Loss": [],
    "Test Accuracy": []
}

# 超参数
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
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

# 构建数据加载器
trainDataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# 初始化模型
net = Net(KEEP_PROB_RATE).to(device)
logger.info("Model initialized.")

# 损失函数 & 优化器
lossF = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
# 定义 AdamW 优化器
optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 记录最佳准确率
best_accuracy = 0.0

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

    # 训练过程记录损失和准确率
    history["Train Loss"].append(epochTrainLoss)
    history["Train Accuracy"].append(epochTrainAccuracy)

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
    history["Test Loss"].append(testLoss)
    history["Test Accuracy"].append(testAccuracy)

    logger.info(f"Epoch {epoch}: Train Loss: {epochTrainLoss:.4f}, Train Acc: {epochTrainAccuracy:.4f}, Test Loss: {testLoss:.4f}, Test Acc: {testAccuracy:.4f}")

    # **保存最佳权重**
    if testAccuracy > best_accuracy:
        best_accuracy = testAccuracy
        torch.save(net.state_dict(), "weights/best_mnist_cnn.pth")
        logger.info(f"New best model saved with Test Accuracy: {best_accuracy:.4f}")

# **训练结束后，保存最后一轮权重**
torch.save(net.state_dict(), "weights/last_mnist_cnn.pth")

# 训练完成后绘图
plot_metrics(history, save_path="logs/img/training_curve.png")
logger.info("Training completed. Model and plots saved.")