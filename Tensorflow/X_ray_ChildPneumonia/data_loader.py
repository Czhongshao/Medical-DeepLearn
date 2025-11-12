# 导入相关库
import cv2
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["GRPC_VERBOSITY"] = "ERROR"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 支持负号
# 读取图像
import matplotlib.image as mpimg
# TensorFlow 和 Keras 层、模型、优化和损失
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import * 
from tensorflow.keras.losses import BinaryCrossentropy # 二元交叉熵损失函数
# 初始化内核。LabelEncoder 工具将将文本标签转换为数值标签
from sklearn.preprocessing import LabelEncoder 
# 自适应矩估计优化器
from tensorflow.keras.optimizers import Adam , Adamax
# 预训练模型 Xception
from tensorflow.keras.applications import *
# 早停回调函数。在训练过程中监控验证集的性能，当性能不再提升时提前停止训练
from tensorflow.keras.callbacks import EarlyStopping
import warnings 
warnings.filterwarnings("ignore")


USE_GPU = True 
if USE_GPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 启用成功:", gpus)
    else:
        print("❌ 未检测到 GPU，将使用 CPU。")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("🚫 GPU 计算关闭，使用 CPU。")


# 数据集路径
train_directory = "./data/train"
test_directory = "./data/test"
val_directory = "./data/val"

IMAGE_SIZE = (256, 256)

def load_data():
    print('-' * 20 + 'DATA LOADING' +'-' * 20)

    # 随机展示部分数据
    show_data('TRAIN', train_directory)
    show_data('TEST', test_directory)

    print('-' * 20 + 'MAKE DATASETS' +'-' * 20)

    # 训练集
    print('TRAIN DATASET: ')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=32
    )

    # 验证集
    print('VAL DATASET: ')
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=32
    )

    # 测试集
    print('TEST DATASET: ')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_directory,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=32
    )

    return train_ds, validation_ds, test_ds

def show_data(dataname, directory):
    filepath =[] # 存储图像存放路径
    label = [] # 存储对应标签

    folds = os.listdir(directory)

    for fold in folds:
        f_path = os.path.join(directory, fold)
        imgs = os.listdir(f_path)

        for img in imgs:
            img_path = os.path.join(f_path, img)
            filepath.append(img_path)
            label.append(fold)

    # 链接数据路径和标签
    file_path_series = pd.Series(filepath, name='filepath')
    Label_path_series = pd.Series(label, name='label')
    df= pd.concat([file_path_series, Label_path_series], axis=1) 
    
    # 查看部分数据情况
    print(f'{dataname} data:')
    print(df.sample(5))

# load_data()