from tensorflow.keras.callbacks import EarlyStopping
from src.model_builder import build_model
from src.data_loader import load_data


def train_model():
    train_ds, validation_ds, _ = load_data()
    model = build_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=validation_ds,
        callbacks=[early_stopping]
    )

    return model, history