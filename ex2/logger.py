import logging
import os

def setup_logger(log_dir="logs", log_filename="training.log", log_level=logging.INFO):
    """
    设置日志记录器，记录日志到控制台和文件。

    参数：
    - log_dir: 日志存储目录，默认 "logs"。
    - log_filename: 日志文件名，默认 "training.log"。
    - log_level: 日志级别，默认 `INFO`。
    
    返回：
    - logger: 配置好的日志记录器。
    """
    # 创建日志存储目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_filename)

    # 创建一个日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 创建文件处理器，日志写入文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)

    # 创建控制台处理器，日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger