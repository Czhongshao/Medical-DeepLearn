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


def train_model(train_ds, val_ds, epochs, patience):
    print()
    print('-' * 50 + 'TRAIN MODEL' +'-' * 50)

    print('-' * 30 + 'LOADING BASE MODEL' +'-' * 30)
    # 加载 Xception 基础模型，不带顶部层
    base_model = Xception(weights='./models/pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False, 
                        pooling='avg', 
                        input_shape=(256, 256, 3))

    base_model.trainable = False

    model = Sequential()

    model.add(base_model)

    model.add(BatchNormalization())

    model.add(Dropout(0.45)) 

    model.add(Dense(220, activation='relu'))

    model.add(Dropout(0.25)) 

    model.add(Dense(60,activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    print('-' * 30 + 'FITTING MODEL' +'-' * 30)
    history = fitting_model(model, epochs, patience, train_ds, val_ds)
    
    print('-' * 30 + 'VALIDATION LOSS AND ACCURACY' +'-' * 30)
    val_loss_ac(model, val_ds)

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
                        callbacks=[early_stopping])
    
    return history

def val_loss_ac(model, val):
    # 在验证集上评估模型
    val_loss, val_accuracy = model.evaluate(val)

    # 输出验证集损失与准确率
    print("VAL LOSS: ", val_loss)
    print("VAL AC: ", val_accuracy)



# # 获取验证集最高准确率的 epoch
# best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1

# # 绘制图像
# plt.style.use('seaborn-v0_8-whitegrid')
# # 创建 1 行 2 列子图
# fig, axs = plt.subplots(1, 2, figsize=(16, 5))

# # 绘制训练和验证准确率图像
# axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
# axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
# axs[0].scatter(best_epoch - 1, history.history['val_accuracy'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
# axs[0].set_xlabel('Epoch')
# axs[0].set_ylabel('Accuracy')
# axs[0].set_title('TRAIN and VAL Accuracy')
# axs[0].legend()

# # 绘制训练和验证损失值图像
# axs[1].plot(history.history['loss'], label='Training Loss', color='blue')
# axs[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
# axs[1].scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='green',label=f'Best Epoch: {best_epoch}')
# axs[0].set_xlabel('Epoch')
# axs[0].set_ylabel('Loss')
# axs[1].set_title('TRAIN and VAL Loss')
# axs[1].legend()

# plt.tight_layout()
# # plt.savefig("out_fig/best_epoch.png", dpi=300)
# plt.show()