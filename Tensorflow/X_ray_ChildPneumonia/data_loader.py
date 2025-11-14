import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import tensorflow as tf

# 数据集路径
# train_directory = "./data/train"
# test_directory = "./data/test"
# val_directory = "./data/val"

# IMAGE_SIZE = (256, 256)
# batch_size = 32

def load_data(train_dir, val_dir, test_dir, IMAGE_SIZE, batch_size):
    print()
    print('-' * 50 + 'DATA LOADER' +'-' * 50)
    
    print('-' * 30 + 'DATA LOADING' +'-' * 30)

    # 随机展示部分数据
    show_data('TRAIN', train_dir)
    show_data('TEST', test_dir)
    show_data('VAL', val_dir)

    print('-' * 30 + 'MAKE DATASETS' +'-' * 30)

    # 训练集
    print('TRAIN DATASET: ')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=batch_size
    )

    # 验证集
    print('VAL DATASET: ')
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=batch_size
    )

    # 测试集
    print('TEST DATASET: ')
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=batch_size
    )

    # 输出训练集形状
    print('-' * 30 + 'CHECK DATASHAPE' +'-' * 30)
    check_traindatashpae(train_ds)

    # 设计标签分类
    class_labels = get_labels(train_ds, test_ds, val_ds)

    # 归一化
    print('-' * 30 + 'PIX NORMALIZATION' +'-' * 30)
    train_ds = pix_normalization(train_ds)
    val_ds = pix_normalization(val_ds)
    test_ds = pix_normalization(test_ds)
    print('NORMALIZATION FINISHED')

    return train_ds, val_ds, test_ds, class_labels

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

def get_labels(train_ds, test_ds, val_ds):
    # 提取 labels
    train_labels = train_ds.class_names
    test_labels = test_ds.class_names
    val_labels = val_ds.class_names

    # 定义类别标签
    class_labels = ['NORMAL', 'PNEUMONIA'] 

    # 初始化标签，将文本标签转换为数值标签
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)

    # 转化标签
    train_labels_encoded = label_encoder.transform(train_labels)
    validation_labels_encoded = label_encoder.transform(val_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    return class_labels

def check_traindatashpae(ds):
    for image_batch, labels_batch in ds:
        print("Shape of X_train: ", image_batch.shape)
        print("Shape of y_train: ", labels_batch.shape)
        break

def pix_normalization(ds):
    return ds.map(lambda x, y: (x / 255.0, y))


# load_data()