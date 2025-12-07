import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import base64

class EmotionDetector:
    def __init__(self, model_path, cascade_path):
        self.model_path = model_path
        self.cascade_path = cascade_path
        self.model = self._build_model()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.labels = {
            0: "Enojado", 
            1: "Disgustado", 
            2: "Miedo", 
            3: "Feliz", 
            4: "Neutral", 
            5: "Triste", 
            6: "Sorpresa"
        }
        self.load_weights()

    def _build_model(self):
        # Reconstrucción exacta de la arquitectura del script original
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        return model

    def load_weights(self):
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print("[INFO] Pesos de Emoción cargados.")
        else:
            print(f"[ERROR] No se encuentra {self.model_path}")

    def predict_from_base64(self, base64_str):
        try:
            # 1. Decodificar imagen base64
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            img_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 2. Preprocesamiento (Gris y Detección de Rostro)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                return None, 0.0 # No se detectó cara

            # Tomamos la cara más grande
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y + h, x:x + w]
            
            # 3. Preparar para el modelo (48x48)
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            
            # 4. Predicción
            prediction = self.model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            return self.labels[maxindex], confidence

        except Exception as e:
            print(f"Error en predicción visual: {e}")
            return None, 0.0