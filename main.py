import numpy as np
import io
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import uvicorn
from contextlib import asynccontextmanager

# Modelo global
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar la aplicación
    global model
    model = tf.keras.models.load_model('./mnist_model.h5')
    print("Modelo cargado correctamente")

    yield  # Este punto es donde la aplicación está funcionando

    # Código que se ejecuta al cerrar la aplicación
    print("Liberando recursos del modelo...")
    # Aquí podrías liberar recursos si fuera necesario

# Crear la aplicación FastAPI con el lifespan
app = FastAPI(
    title="MNIST Digit Classifier API",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "MNIST Digit Classifier API",
            "instructions": "POST an image to /predict to get a digit prediction",
            "test_image": "POST an image to /test-image to test with MNIST dataset images"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para predecir dígitos a partir de imágenes subidas por el usuario.
    Aplica un preprocesamiento completo para adaptar la imagen al formato esperado
    por el modelo (escala de grises, 28x28, inversión de colores para fondo blanco).
    """
    # Leer imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocesar imagen
    image = image.convert("L")  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar

    # Invertir colores si es necesario (para imágenes con fondo blanco)
    image = ImageOps.invert(image)

    # Convertir a array y normalizar
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Realizar predicción
    predictions = model.predict(img_array)

    # Interpretar resultados
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": {str(i): float(prob) for i, prob in enumerate(predictions[0])}
    }


@app.post("/test-image/")
async def test_image(file: UploadFile = File(...)):
    """
    Endpoint para probar el modelo con imágenes del dataset MNIST.
    Aplica un preprocesamiento mínimo, asumiendo que las imágenes ya están
    en formato adecuado (escala de grises, 28x28, fondo negro y dígito blanco).
    """
    # Leer imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Convertir a array y normalizar (preprocesamiento mínimo)
    img_array = np.array(image)

    # Si la imagen tiene 3 canales (RGB), convertirla a escala de grises
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]  # Tomamos solo un canal

    # Normalización
    img_array = img_array / 255.0

    # Reshape para modelo
    img_array = img_array.reshape(1, 28, 28, 1)

    # Realizar predicción
    predictions = model.predict(img_array)

    # Interpretar resultados
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": {str(i): float(prob) for i, prob in enumerate(predictions[0])}
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# .\env_cnn\Scripts\activate && python main.py
# run_api.bat