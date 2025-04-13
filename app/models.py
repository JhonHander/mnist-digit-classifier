import tensorflow as tf
import numpy as np

class MNISTModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print("Modelo cargado correctamente")

    def predict(self, img_array: np.ndarray):
        predictions = self.model.predict(img_array)
        return predictions