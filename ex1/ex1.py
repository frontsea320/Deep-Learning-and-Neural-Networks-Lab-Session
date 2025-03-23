import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, optimizers
from plot_metrics import plot_metrics
from save_weights import save_weights

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 准备数据
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x / 255.0
    x_test = x_test / 255.0
    return (x, y), (x_test, y_test)

# 定义矩阵乘法类
class Matmul:
    def __init__(self):
        self.mem = {}
    
    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem = {"x": x, "W": W}
        return h
    
    def backward(self, grad_y):
        x = self.mem["x"]
        W = self.mem["W"]
        grad_x = np.matmul(grad_y, W.T)
        grad_W = np.matmul(x.T, grad_y)
        return grad_x, grad_W

# 定义 ReLU 类
class Relu:
    def __init__(self):
        self.mem = {}
    
    def forward(self, x):
        self.mem["x"] = x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        x = self.mem["x"]
        return (x > 0).astype(np.float32) * grad_y

# 定义 Softmax 类
class Softmax:
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12
    
    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)  # 防止溢出
        x_exp = np.exp(x)
        denominator = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (denominator + self.epsilon)
        self.mem["out"] = out
        return out
    
    def backward(self, grad_y):
        s = self.mem["out"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)
        tmp = np.squeeze(tmp, axis=1)
        return -tmp + grad_y * s

# 定义交叉熵类
class CrossEntropy:
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
    
    def forward(self, x, labels):
        log_prob = np.log(x + self.epsilon)
        out = np.mean(np.sum(-log_prob * labels, axis=1))
        self.mem["x"] = x
        return out
    
    def backward(self, labels):
        x = self.mem["x"]
        return -1 / (x + self.epsilon) * labels

# 定义模型
class MyModel:
    def __init__(self, pretrained_dir=None):
        # 初始化权重
        self.W1 = np.random.normal(size=[28 * 28 + 1, 100])
        self.W2 = np.random.normal(size=[100, 66])
        self.W3 = np.random.normal(size=[66, 10])
        self.W_skip = np.random.normal(size=[100, 66])

        self.mul_h1 = Matmul()
        self.relu1 = Relu()
        self.mul_h2 = Matmul()
        self.relu2 = Relu()
        self.mul_h3 = Matmul()
        self.softmax = Softmax()
        self.cross_en = CrossEntropy()

        # 如果指定了预训练权重目录，加载权重
        if pretrained_dir is not None:
            self.load_weights(pretrained_dir)
        
    def load_weights(self, weights_dir):
        try:
            self.W1 = np.load(os.path.join(weights_dir, "W1.npy"))
            self.W2 = np.load(os.path.join(weights_dir, "W2.npy"))
            self.W3 = np.load(os.path.join(weights_dir, "W3.npy"))
            self.W_skip = np.load(os.path.join(weights_dir, "W_skip.npy"))
            print(f"Pretrained weights loaded from '{weights_dir}' successfully!")
        except FileNotFoundError as e:
            print(f"Error: {e}. Using randomly initialized weights.")

    
    def forward(self, x, labels):
        x = x.reshape(-1, 28 * 28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)

        self.h1 = self.mul_h1.forward(x, self.W1)
        self.h1_relu = self.relu1.forward(self.h1)

        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_relu = self.relu2.forward(self.h2)

        self.h1_skip = np.matmul(self.h1_relu, self.W_skip)
        self.h2_res = self.h2_relu + self.h1_skip

        self.h3 = self.mul_h3.forward(self.h2_res, self.W3)
        self.h3_soft = self.softmax.forward(self.h3)
        self.loss = self.cross_en.forward(self.h3_soft, labels)
    
    def backward(self, labels):
        self.loss_grad = self.cross_en.backward(labels)
        self.h3_soft_grad = self.softmax.backward(self.loss_grad)

        self.h3_grad, self.W3_grad = self.mul_h3.backward(self.h3_soft_grad)

        self.h2_res_grad = self.h3_grad
        self.h2_relu_grad = self.h2_res_grad
        self.h1_skip_grad = np.matmul(self.h2_res_grad, self.W_skip.T)

        self.h1_relu_grad = self.relu1.backward(self.h1_skip_grad)

        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_relu_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)
        self.W_skip_grad = np.matmul(self.h1_relu.T, self.h2_res_grad)

# 计算准确率
def compute_accuracy(prob, labels):
    predictions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predictions == truth)

def train_one_step(model, x, y, learning_rate):
    model.forward(x, y)
    model.backward(y)

    # 更新权重时使用自适应学习率
    model.W1 -= learning_rate * model.W1_grad
    model.W2 -= learning_rate * model.W2_grad
    model.W3 -= learning_rate * model.W3_grad
    model.W_skip -= learning_rate * model.W_skip_grad

    loss = model.loss
    accuracy = compute_accuracy(model.h3_soft, y)
    return loss, accuracy

# 测试模型
def test(model, x, y):
    model.forward(x, y)
    loss = model.loss
    accuracy = compute_accuracy(model.h3_soft, y)
    return loss, accuracy

# 训练模型
if __name__ == "__main__":
    # 当且仅当你直接执行 model.py 时，这段代码才会运行。
    model = MyModel()
    train_data, test_data = mnist_dataset()
    train_label = np.zeros(shape=[train_data[0].shape[0], 10])
    test_label = np.zeros(shape=[test_data[0].shape[0], 10])
    train_label[np.arange(train_data[0].shape[0]), train_data[1]] = 1
    test_label[np.arange(test_data[0].shape[0]), test_data[1]] = 1

    initial_lr = 5e-4  # 初始学习率
    loss_list, accuracy_list = [], []
    for epoch in range(200):
    # 计算当前 epoch 的学习率（指数衰减）
        learning_rate = initial_lr #* (0.95 ** epoch)
        loss, accuracy = train_one_step(model, train_data[0], train_label, learning_rate=learning_rate)

        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f'Epoch {epoch}: Loss {loss:.4f}; Accuracy {accuracy:.4f}; LR {learning_rate:.6f}')

    plot_metrics(loss_list, accuracy_list, save_dir="homework/ex1")

    loss, accuracy = test(model, test_data[0], test_label)
    print(f'Test Loss {loss}; Accuracy {accuracy}')

    save_weights(model, save_dir="homework/ex1/weight")
    print('权重文件已保存到 homework/ex1/weight')