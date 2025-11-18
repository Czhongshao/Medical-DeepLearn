import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 支持负号
import seaborn as sns


def val_loss_ac(model, val):
    print()
    print('-' * 50 + 'VALIDATION METRICS' +'-' * 50)

    print('-' * 30 + 'PRINT VALIDATION METRICS' +'-' * 30)

    # 在验证集上评估模型
    results = model.evaluate(val, verbose=0)
    metrics_names = model.metrics_names
    print("Validation Metrics:")
    for name, value in zip(metrics_names, results):
        print(f"{name.upper()}: {value:.4f}")


def loss_ac_plot(history):
    print('-' * 30 + 'PRINT LOSS AND AC PLOT' + '-' * 30)

    # 兼容 History
    if hasattr(history, "history"):
        history = history.history

    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    prec = history.get('precision', [])
    val_prec = history.get('val_precision', [])
    rec = history.get('recall', [])
    val_rec = history.get('val_recall', [])
    auc = history.get('auc', [])
    val_auc = history.get('val_auc', [])

    epochs = range(1, len(loss) + 1)
    # 获取验证集最高准确率的 epoch
    best_epoch = val_acc.index(max(val_acc)) + 1

    # 绘制图像
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 损失 Loss
    axs[0, 0].plot(epochs, loss, label='Training Loss', color='blue')
    axs[0, 0].plot(epochs, val_loss, label='Validation Loss', color='red')
    axs[0, 0].scatter(best_epoch, val_loss[best_epoch-1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].legend()

    # 准确率 Accuracy 
    axs[0, 1].plot(epochs, acc, label='Training Accuracy', color='blue')
    axs[0, 1].plot(epochs, val_acc, label='Validation Accuracy', color='red')
    axs[0, 1].scatter(best_epoch, val_acc[best_epoch-1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Training and Validation Accuracy')
    axs[0, 1].legend()

    # 精确率与回归率 Precision and Recall
    axs[1, 0].plot(epochs, prec, label='Training Precision', color='purple')
    axs[1, 0].plot(epochs, val_prec, label='Validation Precision', color='orange')
    axs[1, 0].plot(epochs, rec, label='Training Recall', color='green')
    axs[1, 0].plot(epochs, val_rec, label='Validation Recall', color='brown')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].set_title('Precision and Recall')
    axs[1, 0].legend()

    # AUC
    axs[1, 1].plot(epochs, auc, label='Training AUC', color='darkblue')
    axs[1, 1].plot(epochs, val_auc, label='Validation AUC', color='darkred')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('AUC')
    axs[1, 1].set_title('Training and Validation AUC')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("./output/train_metrics.png")
    plt.close()
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

