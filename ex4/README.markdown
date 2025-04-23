# 基因表达量预测深度学习项目（模型代码coming soon）

## ✨ 项目概述

本项目开发了深度学习模型，用于从DNA序列预测基因表达量（以TPM，百万转录本为单位）。项目实现了两种主要模型：

1. **BasenjiModel**：一种受Basenji架构启发的卷积神经网络（CNN），结合残差连接和空洞卷积，旨在捕捉DNA序列中的长距离依赖关系。
2. **QuaternionBasenjiModel**：一种新颖的模型，引入四元数卷积层，通过将DNA序列建模为四元数表示来增强特征提取，可能捕获更复杂的模式。

两种模型均基于包含DNA序列及其对应TPM值的Excel数据集（`dataset.xlsx`）进行训练。项目包括完整的数据预处理、模型训练、评估和可视化脚本。

## ✨ 功能特点

- **数据预处理**：
  - 从Excel文件加载DNA序列和TPM值。
  - 对DNA序列（A、C、G、T）进行one-hot编码。
  - 支持预定义的数据集划分（`train`、`valid`、`test`）或自动划分（80%训练，20%测试）。
  - 对基因表达量应用log2(TPM + 1)变换以稳定方差。

- **模型**：
  - **BasenjiModel**：结合卷积块、卷积塔、空洞残差块和最终线性层，使用自定义GELU激活函数和支持“same”填充的最大池化。
  - **QuaternionBasenjiModel**：采用四元数卷积和线性层，包含空洞残差块和全局平均池化层，通过计算四元数模进行最终预测。

- **训练与评估**：
  - 训练脚本（`train.py`、`qtrain.py`、`train_model.py`）支持Adam优化器和自定义损失函数。
  - **QuaternionRegressionLoss**：为四元数模型设计的自定义损失函数，结合模差和余弦相似度。
  - 评估指标包括均方误差（MSE）损失和皮尔逊相关系数（PCC）。
  - 提供训练历史（损失和PCC）以及测试集预测（散点图）的可视化。

- **依赖项**：
  - Python 3.8+
  - PyTorch
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - TQDM
  - SciPy
  - Einops
  - Torchsummary

## ⚙️ 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/your-username/gene-expression-prediction.git
   cd gene-expression-prediction
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows系统：venv\Scripts\activate
   ```

3. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

   示例`requirements.txt`：
   ```
   torch>=1.9.0
   pandas>=1.3.0
   numpy>=1.19.0
   scikit-learn>=0.24.0
   matplotlib>=3.4.0
   tqdm>=4.62.0
   scipy>=1.7.0
   einops>=0.4.0
   torchsummary>=1.5.0
   ```

## 数据集

数据集（`dataset.xlsx`）需包含以下列：
- `Geneid`：基因唯一标识符。
- `sequence`：DNA序列（A、C、G、T的字符串）。
- `TPM`：基因表达量（数值）。
- `dataset`（可选）：划分标志（`train`、`valid`、`test`）。若无此列，数据集将自动划分。

运行以下命令准备数据集：
```bash
python data.py
```
此脚本生成`processed_data.pkl`，包含预处理的训练、验证和测试集。

## 🚀使用方法

### 🧠 训练Basenji模型
运行标准CNN模型的训练脚本：
```bash
python train.py --batch_size 32 --epochs 10 --lr 0.001 --data_path processed_data.pkl
```
此命令训练`BasenjiModel`，保存最佳模型（`best_model.pth`），并生成：
- `test_scatter.png`：测试集预测值与真实值的散点图。
- `training_history.png`：训练/验证损失和验证PCC的曲线图。
- `training_results.pkl`：训练历史和测试结果。

### 🧠 训练四元数模型
运行四元数模型的训练脚本：
```bash
python qtrain.py
```
此命令训练`QuaternionBasenjiModel`，保存最佳模型（`best_qmodel.pth`），并生成：
- `ori_plot.png`：测试集预测值与真实值的散点图。
- `training_log.txt`：训练和验证指标的日志。
- `training_validation_loss_pcc.png`（通过`qloss.py`生成）：训练损失和验证PCC的曲线图。

可视化训练日志：
```bash
python qloss.py
```

### 🧠 测试四元数模型
在测试集上评估四元数模型：
```bash
python qtest.py
```
此命令加载训练好的模型（`best_qmodel.pth`），并生成：
- `test_plot.png`：测试集预测值与真实值的散点图。
- 控制台输出测试损失和PCC。

## 📁 项目结构

```
gene-expression-prediction/
│
├── data.py                # 数据预处理和划分
├── model.py               # BasenjiModel定义
├── qlayer.py              # 四元数卷积和线性层
├── qmodel.py              # QuaternionBasenjiModel定义
├── qdata.py               # 四元数模型的数据加载和数据集类
├── qtrain.py              # 四元数模型的训练脚本
├── qtest.py               # 四元数模型的测试脚本
├── qloss.py               # 训练日志可视化
├── train.py               # BasenjiModel的训练脚本
├── train_model.py         # BasenjiModel的替代训练脚本
├── dataset.xlsx           # 输入数据集
├── processed_data.pkl     # 预处理数据（由data.py生成）
├── best_model.pth         # 训练好的BasenjiModel权重
├── best_qmodel.pth        # 训练好的QuaternionBasenjiModel权重
├── requirements.txt       # Python依赖
└── README.md              # 本文件
```

## 结果展示

- **BasenjiModel**：在测试集上获得较高的PCC，散点图显示预测值与真实值的相关性。
- **QuaternionBasenjiModel**：通过四元数表示可能提升特征提取能力，性能通过PCC和自定义损失评估。

示例输出：
- 训练历史图（`training_history.png`、`training_validation_loss_pcc.png`）展示损失收敛和PCC提升。
- 散点图（`test_scatter.png`、`ori_plot.png`、`test_plot.png`）可视化预测准确性。

## 📜 未来改进

- 在`train_model.py`中实现早停机制以防止过拟合。
- 探索模型超参数调优（如学习率、批量大小、残差块数量）。
- 添加交叉验证以更稳健地评估模型性能。
- 研究其他基于四元数的架构或损失函数。
- 优化计算效率以处理更大规模数据集。

## 贡献指南

欢迎贡献代码！请按照以下步骤：
1. Fork本仓库。
2. 创建功能分支（`git checkout -b feature/your-feature`）。
3. 提交更改（`git commit -m 'Add your feature'`）。
4. 推送分支（`git push origin feature/your-feature`）。
5. 提交Pull Request。

## 许可证

本项目采用MIT许可证，详情见`LICENSE`文件。

## 联系方式

如有问题或建议，请提交Issue。

🎉 **每个渺小的理由 都困住自由**