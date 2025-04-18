# MNIST Digit Classifier

Este proyecto implementa un clasificador de dígitos escritos a mano utilizando el dataset MNIST y redes neuronales convolucionales. Incluye un notebook de análisis y entrenamiento, así como una API REST para consumir el modelo entrenado.

## 📋 Descripción

El proyecto consta de dos componentes principales:
- Un notebook de análisis exploratorio de datos (EDA) y entrenamiento del modelo CNN
- Una API REST construida con FastAPI para hacer predicciones con el modelo entrenado

## 🛠️ Requisitos

- Python 3.8+
- Bibliotecas requeridas (instalables vía `pip install -r requirements.txt`):
  - TensorFlow
  - FastAPI
  - Pillow
  - NumPy
  - uvicorn
  - python-multipart

## 🚀 Configuración y Ejecución

### Opción 1: Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv env
   # En Windows
   .\env\Scripts\activate
   # En Linux/Mac
   source env/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Entrenar el modelo (opcional):**
   
   Si deseas entrenar el modelo desde cero, ejecuta el notebook `notebooks/CNN_numbers_clasificator.ipynb`.
   Este notebook también genera imágenes de prueba en el directorio `test_images/`.

5. **Ejecutar la API:**
   ```bash
   python main.py
   ```
   
   La API estará disponible en http://localhost:8000

### Opción 2: Ejecución con Docker

1. **Construir la imagen Docker:**
   ```bash
   docker build -t mnist-classifier .
   ```

2. **Ejecutar el contenedor:**
   ```bash
   docker run -p 8000:8000 mnist-classifier
   ```

   La API estará disponible en http://localhost:8000

## 📊 Análisis de Datos y Entrenamiento

El notebook `notebooks/CNN_numbers_clasificator.ipynb` contiene:

1. **Carga y exploración del dataset MNIST**
2. **Análisis exploratorio de datos (EDA)**
   - Distribución de clases
   - Visualización de ejemplos
   - Análisis de intensidad de píxeles
3. **Preprocesamiento de datos**
   - Normalización
   - Transformación de dimensiones
   - One-hot encoding
4. **Arquitectura del modelo CNN**
   - Capas convolucionales
   - Capas de pooling
   - Dropout para reducir overfitting
5. **Entrenamiento del modelo**
6. **Evaluación y visualización de resultados**
7. **Guardado del modelo**

## 🔌 Uso de la API

La API expone dos endpoints para la predicción de dígitos:

### `/predict`

Para imágenes subidas por el usuario, con preprocesamiento completo para normalizar y adaptar cualquier imagen.

**Ejemplo con curl:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/ruta/a/tu/imagen.png'
```

### `/test-image`

Para imágenes del dataset MNIST que encuentras en `test_images/`, 

**Ejemplo con Postman:**

1. Selecciona el método POST
2. Ingresa la URL: `http://localhost:8000/predict/` o `http://localhost:8000/test-image/`
3. En la pestaña "Body", selecciona "form-data"
4. Añade una key llamada "file" (tipo File)
5. Selecciona tu imagen y envía la solicitud

## 📂 Estructura del Proyecto

```
mnist-digit-classifier/
├── notebooks/
│   └── CNN_numbers_clasificator.ipynb  # Notebook de análisis y entrenamiento
├── test_images/                        # Imágenes de prueba (generadas por el notebook)
├── main.py                             # Aplicación FastAPI
├── requirements.txt                    # Dependencias completas
├── requirements.prod.txt               # Dependencias mínimas para el contenedor
├── mnist_model.h5                      # Modelo entrenado
├── Dockerfile                          # Configuración para Docker
├── .dockerignore                       # Archivos a ignorar en la imagen Docker
└── README.md                           # Este archivo
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerencias y mejoras.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.