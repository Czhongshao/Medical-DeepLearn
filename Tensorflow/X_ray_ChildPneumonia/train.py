import os
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
# 评价指标
from tensorflow.keras.metrics import Precision, Recall, AUC


def train_model(train_ds, val_ds, epochs, patience):
    print()
    print('-' * 50 + 'TRAIN MODEL' +'-' * 50)

    print('-' * 30 + 'LOADING BASE MODEL' +'-' * 30)
    # 加载 Xception 基础模型，不带顶部层
    base_model = Xception(weights='./models/pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False, 
                        pooling='avg',  # 最后一层卷积层后添加全局平均池化
                        input_shape=(256, 256, 3))  # （高，宽，通道）

    base_model.trainable = False  # 冻结基础模型权重，迁移模型的基础步骤

    model = Sequential()  # 建立一个空模型

    model.add(base_model)

    model.add(BatchNormalization())  # 对 Xception 输出归一化

    model.add(Dropout(0.45))  # 随机舍弃45%神经元

    model.add(Dense(220, activation='relu'))  # 全连接层的高维输入 --> 低维。 ReLU 激活函数: f(x) = max(0, x)

    model.add(Dropout(0.25))  # 隐藏层后随机舍弃25%神经元

    model.add(Dense(60,activation='relu'))

    model.add(Dense(1, activation='sigmoid'))  # 输出一个神经元，对应二分类概率。调用sigmoid函数。

    model.compile(
        optimizer=Adamax(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=[
            'accuracy',  # 准确率
            Recall(name='recall'),  # 回报率 
            Precision(name='precision'),  # 精确率
            AUC(name='auc')  # 综合评价
        ]
    )

    print("-" * 30 + 'MODEL SUMMARY' + '-' * 30)
    model.summary()

    print('-' * 30 + 'FITTING MODEL' +'-' * 30)
    history = fitting_model(model, epochs, patience, train_ds, val_ds)

    return model, history

def fitting_model(model, epochs, patience, train_ds, val_ds):
    # 早停回调
    early_stopping = EarlyStopping(monitor='val_loss',
                                patience=patience,
                                restore_best_weights=True)

    # 拟合模型
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=val_ds,
                        callbacks=[early_stopping], 
                        verbose = 1)
    
    return history