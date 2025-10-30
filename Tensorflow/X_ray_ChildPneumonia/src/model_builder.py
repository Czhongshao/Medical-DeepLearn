from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adamax


def build_model():
    # 加载 Xception 基础模型，不带顶部层
    base_model = Xception(
        weights='models/pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False,
        pooling='avg',
        input_shape=(256, 256, 3)
    )
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(BatchNormalization())
    model.add(Dropout(0.45))
    model.add(Dense(220, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model