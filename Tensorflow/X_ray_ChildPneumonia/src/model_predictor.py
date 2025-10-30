import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import load_model


def predict_model():
    model = load_model('models/final/CNN_ChildPneumonia_based_on_Xception.keras')
    test_directory_path = "data/test/PNEUMONIA"
    image_files = os.listdir(test_directory_path)[:5]

    fig, axs = plt.subplots(1, len(image_files), figsize=(15, 5))

    for i, image_file in enumerate(image_files):
        img_path = os.path.join(test_directory_path, image_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img_array = img.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        actual_prediction = (predictions > 0.5).astype(int)

        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].axis('off')
        if actual_prediction[0][0] == 0:
            predicted_label = 'Normal'
        else:
            predicted_label = 'PNEUMONIA'
        axs[i].set_title(f'预测标签: {predicted_label}')

    plt.tight_layout()
    plt.savefig('out_fig/result_actual_prediction.png', dpi=300)
    plt.show()

