import serial
import time
import numpy as np
import tensorflow as tf
from scipy.signal import resample
from colorama import init, Fore, Style
import sys

# --- CONFIGURACIÓN ---
SERIAL_PORT = 'COM9'       # <--- ¡CONFIRMA QUE SIGUE SIENDO ESTE!
BAUD_RATE = 921600
# CAMBIO AQUÍ: Poner el nombre exacto de tu archivo
MODEL_FILE = 'abl_dl_ecg_only.keras'

# Parámetros del Modelo (Deben coincidir con el entrenamiento)
WINDOW_SECONDS = 60        # Ventana de tiempo a analizar
MODEL_FREQ = 256           # Frecuencia con la que se entrenó la IA
MODEL_INPUT_SIZE = WINDOW_SECONDS * MODEL_FREQ # 15360 muestras

# Parámetros de Entrada (Lo que manda el ESP32)
# El ESP32 manda 1 de cada 3 muestras de 700Hz -> ~233.33 Hz
ESP_FREQ = 700 / 3 
SAMPLES_NEEDED = int(WINDOW_SECONDS * ESP_FREQ) # Aprox 14000 muestras

# Inicializar colores consola
init()

def print_status(msg, color=Fore.WHITE):
    print(f"{color}{msg}{Style.RESET_ALL}")

# --- CARGA DEL MODELO ---
print_status("--- CARDIOCALM AI BACKEND ---", Fore.CYAN)
print_status(f"[INIT] Cargando modelo: {MODEL_FILE}...", Fore.YELLOW)
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print_status("[OK] Modelo cargado en memoria.", Fore.GREEN)
except Exception as e:
    print_status(f"[ERROR] No se pudo cargar el modelo: {e}", Fore.RED)
    sys.exit()

# --- CONEXIÓN SERIAL ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print_status(f"[CONN] Conectado a {SERIAL_PORT} @ {BAUD_RATE}", Fore.GREEN)
    # Limpiar buffer inicial
    ser.reset_input_buffer()
except Exception as e:
    print_status(f"[ERROR] Fallo conexión Serial: {e}", Fore.RED)
    sys.exit()

# Buffer de datos
data_buffer = []
clases = ['Baseline', 'Stress', 'Amusement', 'Meditation']

print_status(f"\n[INFO] Escuchando corazón... (Necesito {WINDOW_SECONDS}s de datos)", Fore.CYAN)

while True:
    try:
        # Leer línea del ESP32
        line = ser.readline().decode('utf-8', errors='replace').strip()
        
        # El formato esperado es: E:123.45,P:0,M:0.12
        if line.startswith("E:"):
            parts = line.split(',')
            try:
                # Extraer valor ECG
                ecg_val = float(parts[0].split(':')[1])
                data_buffer.append(ecg_val)
                
                # Feedback de progreso (cada 1000 muestras)
                if len(data_buffer) % 1000 == 0:
                    progreso = (len(data_buffer) / SAMPLES_NEEDED) * 100
                    sys.stdout.write(f"\rRecopilando: [{('#' * int(progreso/5)).ljust(20)}] {int(progreso)}%")
                    sys.stdout.flush()

                # --- MOMENTO DE LA INFERENCIA ---
                if len(data_buffer) >= SAMPLES_NEEDED:
                    print("\n\n>>> PROCESANDO VENTANA...", end="")
                    
                    # 1. Convertir a Numpy
                    signal = np.array(data_buffer)
                    
                    # 2. Resampling (Adaptar 233Hz a 256Hz)
                    # El modelo necesita exactamente 15360 puntos
                    signal_resampled = resample(signal, MODEL_INPUT_SIZE)
                    
                    # 3. Normalización Z-Score (CRÍTICO: Igual que en el notebook DL)
                    # (x - media) / desviacion
                    sig_mean = np.mean(signal_resampled)
                    sig_std = np.std(signal_resampled)
                    if sig_std == 0: sig_std = 1 # Evitar div/0
                    
                    signal_norm = (signal_resampled - sig_mean) / sig_std
                    
                    # 4. Dar forma de Tensor (Batch, Pasos, Canales) -> (1, 15360, 1)
                    input_tensor = signal_norm.reshape(1, MODEL_INPUT_SIZE, 1)
                    
                    # 5. Predicción
                    t0 = time.time()
                    pred = model.predict(input_tensor, verbose=0)[0]
                    t1 = time.time()
                    
                    # 6. Interpretar resultados
                    idx_max = np.argmax(pred)
                    confianza = pred[idx_max]
                    resultado = clases[idx_max]
                    
                    # 7. Mostrar Resultado
                    color_res = Fore.WHITE
                    if resultado == 'Stress': color_res = Fore.RED
                    elif resultado == 'Baseline': color_res = Fore.BLUE
                    elif resultado == 'Meditation': color_res = Fore.GREEN
                    
                    print(f" Hecho en {t1-t0:.2f}s")
                    print_status(f"\n================================", color_res)
                    print_status(f" ESTADO: {resultado.upper()}", color_res)
                    print_status(f" Confianza: {confianza:.2%}", Fore.YELLOW)
                    print_status(f"================================\n", color_res)
                    
                    # 8. Reiniciar Buffer (Ventana deslizante o Reset total)
                    # Para demo, hacemos reset total para empezar nueva medición limpia
                    data_buffer = [] 
                    print_status("[INFO] Iniciando nueva medición...", Fore.CYAN)

            except ValueError:
                pass # Ignorar líneas corruptas

    except KeyboardInterrupt:
        print_status("\n[EXIT] Cerrando conexión.", Fore.YELLOW)
        ser.close()
        break