import cv2
import requests
import threading

class RealtimeDigitRecognizer:
    """
    Clase para capturar video en tiempo real, enviar imágenes a una API para predicción
    y mostrar los resultados superpuestos en el video.
    """

    def __init__(self, api_url, roi_size=150, prediction_interval=5):
        """
        Inicializa la clase con los parámetros necesarios.

        Args:
            api_url (str): URL base de la API para realizar predicciones.
            roi_size (int): Tamaño del área de interés (ROI) en píxeles.
            prediction_interval (int): Número de cuadros entre cada predicción.
        """
        self.api_url = api_url
        self.roi_size = roi_size
        self.prediction_interval = prediction_interval
        self.last_predicted_digit = None
        self.last_confidence = None
        self.frame_count = 0

    def send_prediction_request(self, roi):
        """
        Envía una imagen a la API para obtener una predicción.

        Args:
            roi (numpy.ndarray): Región de interés (ROI) extraída del cuadro de video.
        """
        _, img_encoded = cv2.imencode('.jpg', roi)
        try:
            response = requests.post(
                f"{self.api_url}/predict/",
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            )
            if response.status_code == 200:
                result = response.json()
                self.last_predicted_digit = result['predicted_digit']
                self.last_confidence = result['confidence']
                print(f"Predicción: {self.last_predicted_digit} ({self.last_confidence*100:.2f}%)")
            else:
                print("Error en la API:", response.text)
        except requests.RequestException as e:
            print(f"Error al enviar la petición: {e}")

    def process_frame(self, frame):
        """
        Procesa un cuadro de video, extrae la ROI y envía una predicción si corresponde.

        Args:
            frame (numpy.ndarray): Cuadro de video capturado.
        """
        height, width, _ = frame.shape
        roi = frame[height//2-self.roi_size:height//2+self.roi_size,
                    width//2-self.roi_size:width//2+self.roi_size]

        # Enviar predicción cada `prediction_interval` cuadros
        self.frame_count += 1
        if self.frame_count % self.prediction_interval == 0:
            threading.Thread(target=self.send_prediction_request, args=(roi,)).start()

        # Dibujar el rectángulo de la ROI
        cv2.rectangle(frame, 
                      (width//2-self.roi_size, height//2-self.roi_size), 
                      (width//2+self.roi_size, height//2+self.roi_size), 
                      (0, 255, 0), 2)

        # Mostrar la última predicción en el cuadro
        if self.last_predicted_digit is not None and self.last_confidence is not None:
            text = f"Digito: {self.last_predicted_digit} ({self.last_confidence*100:.2f}%)"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def start(self):
        """
        Inicia la captura de video en tiempo real y el procesamiento de cuadros.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo abrir la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el video.")
                break

            # Procesar el cuadro actual
            self.process_frame(frame)

            # Mostrar el video en tiempo real
            cv2.imshow("Video en tiempo real", frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# URL de la API
api_url = "http://localhost:8000"

# Crear una instancia de la clase y ejecutar
recognizer = RealtimeDigitRecognizer(api_url)
recognizer.start()