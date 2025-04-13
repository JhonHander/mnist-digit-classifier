import numpy as np
from PIL import Image, ImageOps
import io

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesa una imagen para que sea compatible con el modelo MNIST.
    Convierte a escala de grises, redimensiona a 28x28 y normaliza.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("L")  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar
    image = ImageOps.invert(image)  # Invertir colores si es necesario
    img_array = np.array(image) / 255.0  # Normalizar
    return img_array.reshape(1, 28, 28, 1)


def preprocess_minimal(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesa una imagen con un enfoque mínimo, asumiendo que ya está en escala de grises,
    tiene un tamaño de 28x28 y un fondo negro con el dígito en blanco.
    """
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image) / 255.0  # Normalizar
    if len(img_array.shape) == 3:  # Si tiene 3 canales (RGB), convertir a escala de grises
        img_array = img_array[:, :, 0]
    return img_array.reshape(1, 28, 28, 1)


def interpret_predictions(predictions: np.ndarray) -> dict:
    """
    Interpreta las predicciones del modelo y devuelve el dígito predicho,
    la confianza y las probabilidades para cada dígito.
    """
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_digit])
    probabilities = {str(i): float(prob) for i, prob in enumerate(predictions[0])}
    
    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities
    }
