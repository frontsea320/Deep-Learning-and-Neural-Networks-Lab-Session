import torch
import torchvision
import sys
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ex2 import Net  # 导入模型
from visualization import visualize_predictions
from logger import setup_logger

# 设置日志记录器
logger = setup_logger(log_dir="logs", log_filename="testing.log")

# 设备选择
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 解析命令行参数
if len(sys.argv) < 2:
    print("Usage: python test.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]  # 从命令行获取模型路径

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载测试数据集
path = './data/'
testData = datasets.MNIST(path, train=False, transform=transform)
testDataLoader = DataLoader(testData, batch_size=100)

# 加载模型
net = Net().to(device)

try:
    net.load_state_dict(torch.load(model_path))
    logger.info(f"Model loaded successfully from {model_path}!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sys.exit(1)

net.eval()

# 计算测试准确率
correct, totalLoss = 0, 0
lossF = torch.nn.CrossEntropyLoss()

logger.info("Starting evaluation...")

with torch.no_grad():
    for testImgs, labels in testDataLoader:
        testImgs, labels = testImgs.to(device), labels.to(device)

        outputs = net(testImgs)
        loss = lossF(outputs, labels)

        predictions = torch.argmax(outputs, dim=1)
        totalLoss += loss.item()
        correct += torch.sum(predictions == labels).item()

# 计算总的测试准确率和损失
testAccuracy = correct / len(testData)
testLoss = totalLoss / len(testDataLoader)

# 记录结果
logger.info(f"Test Loss: {testLoss:.4f}, Test Accuracy: {testAccuracy:.4f}")

# 可视化分类结果
visualize_predictions(net, testDataLoader, device, num_samples=8, save_path="logs/img/predictions.png")
logger.info("Predictions visualization saved to logs/img/predictions.png")

# 输出最终结果
print(f"Test Loss: {testLoss:.4f}, Test Accuracy: {testAccuracy:.4f}")