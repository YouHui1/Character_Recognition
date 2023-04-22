import logging
import os
from config import *

if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists('./param'):
    os.mkdir('./param')

logger = logging.getLogger('log')

def create_log(num):
    # 1、创建一个logger
    logger.setLevel(logging.DEBUG)

    # 2、创建一个handler，用于写入日志文件
    fh = logging.FileHandler('./log/'+model_lst[num]+'.log')
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 3、定义handler的输 出格式（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 4、给handler添加formatter
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 5、给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

