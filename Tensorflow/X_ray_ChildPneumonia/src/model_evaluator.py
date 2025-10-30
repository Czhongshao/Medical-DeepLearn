from src.model_trainer import train_model
from src.data_loader import load_data


def evaluate_model():
    _, _, test_ds = load_data()
    model, history = train_model()

    validation_loss, validation_accuracy = model.evaluate(test_ds)
    print("验证损失：", validation_loss)
    print("验证准确率：", validation_accuracy)

    return model, history

