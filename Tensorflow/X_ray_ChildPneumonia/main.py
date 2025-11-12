import os
from model_evaluator import evaluate_model
from model_predictor import predict_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf 
import warnings
warnings.filterwarnings("ignore")

# 调用 GPU
USE_GPU = True 
if USE_GPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 启用成功: ✅\n", gpus)
    else:
        print("❌ 未检测到 GPU，将使用 CPU。❌\n")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("🚫 GPU 计算关闭，使用 CPU。🚫\n")


def main():
    model, history = evaluate_model()
    model.save('./models/final/py_CNN_ChildPneumonia_based_on_Xception.keras')
    predict_model()


if __name__ == "__main__":
    main()

