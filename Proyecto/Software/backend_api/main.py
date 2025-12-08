from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from emotion_engine import EmotionDetector
from pydantic import BaseModel
import numpy as np
from scipy.signal import resample
from model_loader import ModelHandler
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# --- 1. CONFIGURACIÓN ---
MODEL_PATH = "abl_dl_ecg_only.keras"
FACE_MODEL_PATH = "model.h5"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
KEY_PATH = "serviceAccountKey.json"
TARGET_SIZE = 15360 

app = FastAPI(title="CardioCalm AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. GESTOR DE WEBSOCKETS ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_ecg(self, data_point: float):
        for connection in self.active_connections:
            try:
                await connection.send_json({"ecg": data_point})
            except Exception:
                pass

manager = ConnectionManager()

# --- 3. INICIALIZAR FIREBASE ---
db = None 
if not os.path.exists(KEY_PATH):
    print(f"[ALERTA] No se encuentra {KEY_PATH}. El guardado en la nube NO funcionará.")
else:
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[INIT] Conexión a Firebase Firestore exitosa.")
    except Exception as e:
        print(f"[ERROR] Fallo al conectar Firebase: {e}")

# --- 4. CARGAR IA ---
ai_ecg = ModelHandler(MODEL_PATH)
ai_face = EmotionDetector(FACE_MODEL_PATH, FACE_CASCADE_PATH)
ecg_classes = ['Baseline', 'Stress', 'Amusement', 'Meditation']

# --- ESQUEMAS DE DATOS ---
class ECGRequest(BaseModel):
    data: list[float]
    fs: int = 233
    userId: str = "anonymous"

class StreamPacket(BaseModel):
    val: float
    
class ImageRequest(BaseModel):
    image: str # Base64
    userId: str = "anonymous"

@app.on_event("startup")
def startup_event():
    print("[INIT] Iniciando Motor de ECG...")
    ai_ecg.load()
    print("[INIT] Iniciando Motor de Emociones...")
    # La carga de pesos ya se hace en __init__ de EmotionDetector, pero validamos
    if ai_face.model:
        print("[INFO] IA Facial lista.")

# --- ENDPOINTS ---

# A. WebSocket ECG
@app.websocket("/ws/live-ecg")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# B. Streaming ECG
@app.post("/stream")
async def stream_data(packet: StreamPacket):
    await manager.broadcast_ecg(packet.val)
    return {"status": "ok"}

# C. Predicción ECG (Guarda en Firebase)
@app.post("/predict")
def predict_stress(payload: ECGRequest):
    if ai_ecg.model is None:
        raise HTTPException(status_code=503, detail="IA ECG no disponible")

    try:
        # 1. Procesamiento Matemático
        signal = np.array(payload.data)
        if len(signal) != TARGET_SIZE:
            signal = resample(signal, TARGET_SIZE)
            
        sig_std = np.std(signal)
        if sig_std == 0: sig_std = 1
        signal_norm = (signal - np.mean(signal)) / sig_std
        
        # 2. Inferencia
        input_tensor = signal_norm.reshape(1, TARGET_SIZE, 1)
        pred = ai_ecg.predict(input_tensor)[0]
        idx_max = np.argmax(pred)
        
        result_class = ecg_classes[idx_max]
        confidence = float(pred[idx_max])

        # 3. GUARDADO EN FIREBASE
        cloud_id = None
        if db is not None:
            try:
                target_user_uid = payload.userId
                doc_ref = db.collection('sessions').document()
                doc_data = {
                    'timestamp': datetime.now(),
                    'type': 'ecg',
                    'prediction': result_class,
                    'confidence': confidence,
                    'probabilities': {k: float(v) for k, v in zip(ecg_classes, pred)},
                    'duration_sec': 60,
                    'device_id': 'ESP32_S3_Wearable',
                    'userId': target_user_uid
                }
                doc_ref.set(doc_data)
                cloud_id = doc_ref.id
                print(f"[NUBE] Resultado ECG guardado. ID: {cloud_id} para UID {target_user_uid}")
            except Exception as e:
                print(f"[ERROR NUBE] No se pudo guardar: {e}")

        return {
            "status": "success",
            "prediction": result_class,
            "confidence": confidence,
            "cloud_id": cloud_id
        }

    except Exception as e:
        print(f"[API ERROR ECG] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# D. Predicción Emoción Facial
@app.post("/predict/emotion")
def predict_emotion(payload: ImageRequest):
    if ai_face is None:
        raise HTTPException(status_code=503, detail="IA Facial no disponible")

    try:
        emotion, conf = ai_face.predict_from_base64(payload.image)

        if emotion is None:
             return {"status": "no_face", "message": "No se detectó rostro"}

        # Guardar en Firebase
        cloud_id = None
        if db is not None:
             try:
                doc_ref = db.collection('emotion_logs').document()
                doc_ref.set({
                    'timestamp': datetime.now(),
                    'emotion': emotion,
                    'confidence': conf,
                    'userId': payload.userId
                })
                cloud_id = doc_ref.id
             except Exception as e:
                 print(f"Error Firebase: {e}")

        return {
            "status": "success",
            "emotion": emotion,
            "confidence": int(conf * 100),
            "cloud_id": cloud_id
        }

    except Exception as e:
        print(f"Error API Emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "firebase": db is not None, 
        "ai_ecg": ai_ecg.model is not None,
        "ai_face_status": "active" if ai_face else "error"
    }