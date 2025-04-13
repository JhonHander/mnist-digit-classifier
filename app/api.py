from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import register_routes
from app.models import MNISTModel
from contextlib import asynccontextmanager

model_handler = MNISTModel('./mnist_model.h5')

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_handler.load_model()
    yield
    print("Liberando recursos del modelo...")

def create_app() -> FastAPI:
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

    # Registrar rutas
    register_routes(app, model_handler)

    return app


# class MNISTAPI:
#     def __init__(self):
#         self.app = FastAPI(
#             title="MNIST Digit Classifier API",
#             lifespan=self.lifespan
#         )
#         self.model_handler = MNISTModel('./mnist_model.h5')
#         self.configure_routes()
#         self.configure_cors()

#     @asynccontextmanager
#     async def lifespan(self, app: FastAPI):
#         self.model_handler.load_model()
#         yield
#         print("Liberando recursos del modelo...")

#     def configure_cors(self):
#         self.app.add_middleware(
#             CORSMiddleware,
#             allow_origins=["*"],
#             allow_methods=["*"],
#             allow_headers=["*"],
#         )

#     def configure_routes(self):
#         @self.app.get("/")
#         def read_root():
#             return {"message": "MNIST Digit Classifier API",
#                     "instructions": "POST an image to /predict to get a digit prediction"}

#         @self.app.post("/predict/")
#         async def predict(file: UploadFile = File(...)):
#             # Aquí iría la lógica de preprocesamiento y predicción
#             pass

# # Crear instancia de la API
# mnist_api = MNISTAPI()
# app = mnist_api.app