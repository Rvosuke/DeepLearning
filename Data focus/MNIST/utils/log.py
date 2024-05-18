import logging


def setup_logger(timestamp):
    """
    设置日志记录器。
    :return:
    """

    logging.basicConfig(filename=f"training_logs_{timestamp}.log",
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()  # 控制台日志处理器
    console.setLevel(logging.INFO)  # 设置日志级别
    console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))  # 设置日志格式
    logging.getLogger('').addHandler(console)  # 将处理器添加到日志记录器
