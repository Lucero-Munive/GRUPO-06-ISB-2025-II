import tensorflow as tf
import os
import sys

class ModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Carga el modelo Keras en memoria si existe el archivo."""
        if not os.path.exists(self.model_path):
            print(f"[ERROR CRÍTICO] No se encuentra el archivo del modelo en: {self.model_path}")
            return None
            
        try:
            print(f"[INFO] Cargando modelo de IA desde: {self.model_path}...")
            # Carga el modelo (puede tardar unos segundos)
            self.model = tf.keras.models.load_model(self.model_path)
            print("[INFO] Modelo InceptionTime cargado y listo para inferencia.")
            return self.model
        except Exception as e:
            print(f"[ERROR] Falló la carga de TensorFlow: {e}")
            return None

    def predict(self, input_tensor):
        """Realiza la predicción de forma segura."""
        if self.model is None:
            raise Exception("El modelo no está cargado.")
        
        # verbose=0 evita que llene la consola de logs por cada predicción
        return self.model.predict(input_tensor, verbose=0)