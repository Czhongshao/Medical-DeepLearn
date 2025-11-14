import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 支持负号
import matplotlib.image as mpimg
import seaborn as sns


def val_loss_ac(model, val):
    print()
    print('-' * 50 + 'VALIDATION LOSS AND ACCURACY' +'-' * 50)

    print('-' * 30 + 'PRINT LOSS AND AC' +'-' * 30)
    # 在验证集上评估模型
    val_loss, val_accuracy = model.evaluate(val)

    # 输出验证集损失与准确率
    print("VAL LOSS: ", val_loss)
    print("VAL AC: ", val_accuracy)


def loss_ac_plot(history):
    print('-' * 30 + 'PRINT LOSS AND AC PLOT' + '-' * 30)

    # 自动兼容 History 对象或字典
    if hasattr(history, "history"):
        history = history.history

    # 避免 KeyError
    acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    # 获取验证集最高准确率的 epoch
    best_epoch = val_acc.index(max(val_acc)) + 1

    # 绘制图像
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    # 训练与验证准确率
    axs[0].plot(acc, label='Training Accuracy', color='blue')
    axs[0].plot(val_acc, label='Validation Accuracy', color='red')
    axs[0].scatter(best_epoch - 1, val_acc[best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('TRAIN and VAL Accuracy')
    axs[0].legend()

    # 训练与验证损失
    axs[1].plot(loss, label='Training Loss', color='blue')
    axs[1].plot(val_loss, label='Validation Loss', color='red')
    axs[1].scatter(best_epoch - 1, val_loss[best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('TRAIN and VAL Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("./output/best_epoch.png")
    # plt.show()

# 绘制图像及其真实和预测标签
def plot_images_with_predictions(model, class_labels, num_images=20, num_images_per_row=5):
    print('-' * 30 + 'PRINT RANDOM PREDICTED 20 IMGS' +'-' * 30)
    import tensorflow as tf
    dataset = tf.keras.utils.image_dataset_from_directory(
        './data/test/',
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )
    # 为一组图像生成预测结果
    predictions = model.predict(dataset)
    # 打乱数据集
    dataset_shuffled = dataset.shuffle(buffer_size=len(dataset))
    
    plt.figure(figsize=(15, 10))
    for i, (images, labels) in enumerate(dataset_shuffled.take(num_images)):
        # 将张量转换为 NumPy 数组
        images = images.numpy()
        
        # 遍历批次中的每张图像
        for j in range(len(images)):
            if i * num_images_per_row + j < num_images:  # 检查图像总数是否超过所需数量
                predicted_class = class_labels[np.argmax(predictions[i * num_images_per_row + j])]
                true_class = class_labels[np.argmax(labels[j])]
                
                plt.subplot(num_images // num_images_per_row + 1, num_images_per_row, i * num_images_per_row + j + 1)
                plt.imshow(images[i].astype("uint8"))
                plt.title(f'TRUE: {true_class}\nPREDICTED: {predicted_class}')
                plt.axis('off')

    plt.tight_layout()
    plt.savefig("./output/random_predicted_20.png")
    # plt.show()

