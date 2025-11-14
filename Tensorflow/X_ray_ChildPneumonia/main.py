import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

import tensorflow as tf 
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_data
from train import train_model
from eval import val_loss_ac, loss_ac_plot, plot_images_with_predictions

# 调用 GPU
def use_gpu(USE_GPU):
    if USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        print('-' * 30 + 'GPU LOADING' +'-' * 30)
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
                name = tf.config.experimental.get_device_details(g)['device_name']
                cc = tf.config.experimental.get_device_details(g)['compute_capability']
                print(f"✅ GPU 就绪：{name}  {cc[0]}.{cc[1]}  (#{gpus.index(g)})")
        else:
            print("❌ 无可用 GPU，fallback 到 CPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("🚫 已强制关闭 GPU")


def main():
    # 数据路径
    train_directory = "./data/train"
    test_directory = "./data/test"
    val_directory = "./data/val"

    # 参数配置
    IMAGE_SIZE = (256, 256)
    batch_size = 32

    # 调用GPU
    use_gpu(True)
    # 载入数据
    train_ds, val_ds, test_ds, class_labels = load_data(train_directory, test_directory, val_directory, IMAGE_SIZE, batch_size)

    need_train_model = str(input("DID YOU NEED TO TRAIN MODEL?(yes/no)"))
    if need_train_model in ['yes', 'y']:
        # 输出模型与训练记录
        model, history = train_model(train_ds, val_ds, epochs=20, patience=5)
    elif need_train_model in ['no', 'n']:
        # 加载模型
        try:
            model = load_model('./models/final/py_CNN_ChildPneumonia_based_on_Xception.keras')
            with open('./models/final/py_trainHistoryDict.txt','rb') as f:
                history=pickle.load(f)
        except: 
            print("YOU DONT HAVE ANY MODELS AND HISTORYS TO USE!!")


    # 验证集损失与准确率
    val_loss_ac(model, val_ds)
    
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    loss_ac_plot(hist_dict)

    plot_images_with_predictions(model, class_labels, num_images=20)


    # 保存训练模型
    if need_train_model in ['yes', 'y']:
        save_model = str(input("DID YOU SAVE THIS MODEL?(yes/no)"))
        if save_model in ['yes', 'y']:
            model.save('./models/final/py_CNN_ChildPneumonia_based_on_Xception.keras')
            with open('./models/final/py_trainHistoryDict.txt', 'wb') as f:
                pickle.dump(history.history, f)
        elif save_model == 'no' or 'n':
            ...


if __name__ == "__main__":
    main()

