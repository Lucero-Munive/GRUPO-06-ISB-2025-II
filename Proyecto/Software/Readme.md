# Arquitectura de Software - CardioCalm AI

Este directorio contiene todo el código fuente del ecosistema **CardioCalm AI**, diseñado para la detección de estrés y ansiedad mediante bioseñales y visión artificial. El sistema sigue una arquitectura *serverless* y orientada a eventos.

## Estructura de Directorios

| Directorio | Descripción | Tecnologías Clave |
| :--- | :--- | :--- |
| `frontend/` | Aplicación Web Progresiva (PWA) | Next.js, React, TailwindCSS, Web Bluetooth API, Recharts |
| `backend_api/` | API REST en la nube | FastAPI, Python, Docker, Google Cloud Run, TensorFlow |
| `modelamiento/` | Entrenamiento de Modelos AI | Jupyter Notebooks, NeuroKit2, Scikit-learn, TensorFlow (InceptionTime) |
| `Datasets/` | Bases de datos crudas y procesadas | WESAD (ECG, EDA, EMG), Scripts de preprocesamiento |
| `backend_python/` | Scripts de utilidad local | Gateway BLE (Python), Pruebas de concepto |

---

## 1. Frontend (`/frontend`)
La interfaz de usuario principal. Es una **Single Page Application (SPA)** construida con Next.js 14.

*   **Funcionalidades Clave**:
    *   **Conexión BLE Directa**: Utiliza la `Web Bluetooth API` para conectarse al ESP32 sin necesidad de servidores intermedios ni aplicaciones nativas. Filtra por nombre (`CardioCalm-Wearable`).
    *   **Visualización en Tiempo Real**: Grafica la señal ECG (700Hz downsampled) usando `Recharts` con ventanas deslizantes para un rendimiento óptimo.
    *   **Gestión de Sesiones**: Autenticación con Firebase Auth y almacenamiento de perfiles.
    *   **Captura de Emociones**: Módulo que captura vídeo de la webcam, redimensiona los frames a 480px (Client-side processing) y los envía al backend para análisis de expresiones faciales.

*   **Despliegue**:
    *   Alojado en **Firebase Hosting**.
    *   Comando de despliegue: `./deploy.ps1` (PowerShell script automatizado).

---

## 2. Backend API (`/backend_api`)
El cerebro del sistema en la nube. Es un servicio contenerizado desplegado en **Google Cloud Run**.

*   **Servicios**:
    *   `POST /predict`: Recibe arrays de señales ECG (ventanas de 60s), las procesa con el modelo **InceptionTime** (.h5) y devuelve la probabilidad de estrés.
    *   `POST /predict/emotion`: Recibe imágenes en Base64, detecta rostros con Haar Cascades (OpenCV) y clasifica emociones básicas (CNN).
    *   **Firestore Integration**: Guarda automáticamente los resultados de cada inferencia asociándolos al ID del usuario.

*   **Infraestructura**:
    *   **Docker**: Entorno aislado basado en Python 3.10-slim.
    *   **Cloud Run**: Escalado automático a cero (Serverless) para optimización de costos.

---

## 3. Modelamiento e Inteligencia Artificial (`/modelamiento`)
Aquí reside la investigación y desarrollo de los algoritmos.

*   **Pipeline de Entrenamiento**:
    1.  **Preprocesamiento**: Limpieza de ruido (filtros Butterworth) y normalización de la señal ECG del dataset WESAD.
    2.  **Extracción de Características (ML Clásico)**: Uso de `NeuroKit2` para obtener HRV (RMSSD, pNN50, LF/HF).
    3.  **Modelos Entrenados**:
        *   **XGBoost / Random Forest**: Basados en características tabulares.
        *   **InceptionTime (Deep Learning)**: CNN 1D entrenada con señales crudas segmentadas. **(Modelo Seleccionado por mejor performance F1 > 96%)**.
*   **Archivos Clave**:
    *   `Procesamiento_ML_DL.ipynb`: Notebook principal de comparación de modelos.
    *   `mejores_modelos/`: Pesos guardados (.h5, .pkl) listos para producción.

---

## 4. Datasets (`/Datasets`)
Gestión del dataset **WESAD** (Wearable Stress and Affect Detection).
*   Contiene los datos crudos de los 15 sujetos.
*   Scripts de análisis exploratorio (EDA) para validar la calidad de la señal y la separabilidad de clases (Estrés vs Línea Base).

---

## Instrucciones de Desarrollo

### Requisitos Previos
*   Node.js 18+
*   Python 3.10+
*   Docker Desktop
*   Google Cloud CLI & Firebase CLI

### Ejecución Local (Frontend)
```bash
cd frontend
npm install
npm run dev
# Acceder a http://localhost:3000
```

### Ejecución Local (Backend)
```bash
cd backend_api
pip install -r requirements.txt
uvicorn main:app --reload
# Acceder a http://localhost:8000/docs
```
