import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_data
from train import train_model

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
    train_ds, val_ds, test_ds,  = load_data(train_directory, test_directory, val_directory, IMAGE_SIZE, batch_size)

    model, history = train_model(train_ds, val_ds, epochs=20, patience=5)

    model.save('./models/final/py_CNN_ChildPneumonia_based_on_Xception.keras')

    # model, history = evaluate_model()
    # model.save('./models/final/py_CNN_ChildPneumonia_based_on_Xception.keras')
    # predict_model()


if __name__ == "__main__":
    main()

