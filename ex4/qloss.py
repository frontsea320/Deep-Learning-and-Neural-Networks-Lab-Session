#     'training_log.txt'
import matplotlib.pyplot as plt

# 定义文件路径
log_file_path = 'training_log.txt' # 请替换为你的日志文件路径

# 初始化存储数据的列表
epochs = []
train_losses = []
valid_pccs = []

# 读取文件并提取数据
with open(log_file_path, 'r') as file:
    for line in file:
        if 'Epoch' in line:  # 如果该行包含'Epoch'
            print(f"Processing line: {line.strip()}")  # 打印每行日志，便于调试
            
            # 按 " - " 分割时间戳和日志信息
            parts = line.split(" - ")
            print(f"Parts: {parts}")  # 打印分割后的部分，查看是否正确分割
            
            if len(parts) > 1:
                epoch_str = parts[1]  # 提取 'Epoch 47: ...'
                epoch_info = epoch_str.split(":")[0].strip()  # 获取 'Epoch 47'
                epoch = int(epoch_info.split()[1].strip())  # 提取数字部分
                print(f"Extracted Epoch: {epoch}")  # 打印提取的epoch信息

                # 提取训练损失
                if 'Train Loss' in line:
                    try:
                        # 提取训练损失，清除多余的逗号
                        train_loss = float(parts[1].split("Train Loss:")[1].split()[0].strip().replace(',', ''))
                        train_losses.append(train_loss)
                        epochs.append(epoch)
                        print(f"Epoch {epoch}: Train Loss = {train_loss}")  # 调试输出
                    except IndexError:
                        print(f"Warning: Failed to extract Train Loss in Epoch {epoch}")
                
                # 提取验证损失和验证PCC
                if 'Valid Loss' in line and 'Valid PCC' in line:
                    try:
                        # 提取验证损失，清除多余的逗号
                        valid_loss = float(parts[1].split("Valid Loss:")[1].split()[0].strip().replace(',', ''))
                        valid_pcc = float(parts[1].split("Valid PCC:")[1].strip())
                        valid_pccs.append(valid_pcc)
                        print(f"Epoch {epoch}: Valid PCC = {valid_pcc}")  # 调试输出
                    except IndexError:
                        print(f"Warning: Failed to extract Valid Loss or PCC in Epoch {epoch}")

# 检查数据是否成功提取
print(f"Extracted {len(epochs)} epochs, {len(train_losses)} train losses, and {len(valid_pccs)} valid PCCs")

# 如果没有数据提取，直接退出
if not epochs or not train_losses or not valid_pccs:
    print("No data extracted, please check the log file format.")
else:
    # 绘制训练损失和验证PCC的折线图
    plt.figure(figsize=(12, 6))

    # 绘制train-loss的折线图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, color='blue', marker='o', label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Train Loss over Epochs')
    plt.grid(True)
    plt.legend()

    # 绘制valid-pcc的折线图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_pccs, color='orange', marker='x', label='Valid PCC')
    plt.xlabel('Epochs')
    plt.ylabel('Valid PCC')
    plt.title('Valid PCC over Epochs')
    plt.grid(True)
    plt.legend()

    # 保存图表到当前路径
    plt.tight_layout()
    plt.savefig('training_validation_loss_pcc.png')  # 保存为png文件
