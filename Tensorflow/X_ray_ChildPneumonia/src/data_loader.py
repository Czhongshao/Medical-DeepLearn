import os
import pandas as pd
import tensorflow as tf

# 数据集路径
train_directory = "data/train"
test_directory = "data/test"
val_directory = "data/val"


def load_data():
    # 训练集
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )

    # 验证集
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )

    # 测试集
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_directory,
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )

    return train_ds, validation_ds, test_ds

