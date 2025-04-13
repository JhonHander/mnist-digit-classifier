from fastapi import FastAPI, File, UploadFile, HTTPException
from app.utils import preprocess_image, preprocess_minimal, interpret_predictions
from app.models import MNISTModel

def register_routes(app: FastAPI, model_handler: MNISTModel):
    @app.get("/")
    def read_root():
        return {"message": "MNIST Digit Classifier API",
                "instructions": "POST an image to /predict to get a digit prediction",
                "test_image": "POST an image to /test-image to test with MNIST dataset images find in test_images/"}

    @app.post("/predict/")
    async def predict(file: UploadFile = File(...)):
        try:
            # Leer y preprocesar la imagen
            image_bytes = await file.read()
            img_array = preprocess_image(image_bytes)

            # Realizar predicción
            predictions = model_handler.predict(img_array)

            # Interpretar resultados
            results = interpret_predictions(predictions)
            return results
            
        except Exception as e:
            raise HTTPException(status_code=400, detail="Error procesando la imagen")

    @app.post("/test-image/")
    async def test_image(file: UploadFile = File(...)):
        try:
            # Leer y preprocesar la imagen pero cargada del MNIST dataset
            image_bytes = await file.read()
            img_array = preprocess_minimal(image_bytes)

            # Realizar predicción
            predictions = model_handler.predict(img_array)

            # Interpretar resultados
            results = interpret_predictions(predictions)
            return results

        except Exception as e:
            raise HTTPException(status_code=400, detail="Error procesando la imagen para /test-image")