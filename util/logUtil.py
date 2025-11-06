import logging #是 Python 内置的日志模块，帮助开发者跟踪程序运行状态、调试错误和记录关键信息。
import os


class Logger(object):

    '''
        日志级别的设置：
            开发环境：DEBUG 或 INFO
            生产环境：INFO 或 WARNING 或 ERROR
    '''
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }
    def __init__(self, root_path, log_name, level='info', fmt='%(asctime)s - %(levelname)s: %(message)s'):
        # 指定日志保存的路径
        self.root_path = root_path

        # 初始logger名称和格式
        self.log_name = log_name

        # 初始格式
        self.fmt = fmt

        # 先声明一个 Logger 对象
        self.logger = logging.getLogger(log_name)

        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))

    '''
         文件日志处理器（FileHandler） ，它会将日志记录写入指定的文件。
    '''
    def get_logger(self):

        path = os.path.join(self.root_path, 'log')
        os.makedirs(path, exist_ok=True)
        file_name = os.path.join(path, self.log_name + '.log')
        fileHandler = logging.FileHandler(file_name, encoding="utf-8", mode="a")

        # 设置日志的输出格式
        formatter = logging.Formatter(self.fmt)
        fileHandler.setFormatter(formatter)

        # 将fileHandler添加到Logger
        self.logger.addHandler(fileHandler)

        return self.logger