import serial
import time
import requests
import json
import sys
import threading
from collections import deque
from colorama import init, Fore, Style

# --- CONFIGURACIÓN ---
SERIAL_PORT = 'COM9'       
BAUD_RATE = 921600
API_BASE = "http://127.0.0.1:8000"
SAMPLES_FOR_AI = 14000 

init()

# Cola para enviar datos de streaming sin bloquear la lectura serial
stream_queue = deque()

def print_status(msg, color=Fore.WHITE):
    print(f"{color}{msg}{Style.RESET_ALL}")

# Hilo secundario que se encarga de mandar datos rápidos a la web
def streamer_thread():
    while True:
        if len(stream_queue) > 0:
            val = stream_queue.popleft()
            try:
                # Enviamos al endpoint de streaming rápido
                requests.post(f"{API_BASE}/stream", json={"val": val}, timeout=0.1)
            except:
                pass # Si falla un punto, no importa
        else:
            time.sleep(0.01)

# Iniciar hilo de streaming
t = threading.Thread(target=streamer_thread, daemon=True)
t.start()

# --- CONEXIÓN SERIAL ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print_status(f"[CONN] Conectado a {SERIAL_PORT}", Fore.GREEN)
    ser.reset_input_buffer()
except Exception as e:
    print_status(f"[ERROR] No hay sensor: {e}", Fore.RED)
    sys.exit()

data_buffer = [] # Buffer para la IA (60s)

print_status("[INFO] Gateway Dual: Streaming en vivo + IA Acumulada", Fore.CYAN)

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        if line.startswith("E:"):
            parts = line.split(',')
            try:
                val = float(parts[0].split(':')[1])
                
                # 1. Enviar a la cola de Streaming (Visualización)
                # Solo mandamos 1 de cada 5 puntos para no saturar HTTP
                # (El ESP32 manda a 233Hz, mandar a 46Hz a la web es suficiente)
                if len(data_buffer) % 5 == 0:
                    stream_queue.append(val)

                # 2. Guardar en Buffer IA (Análisis)
                data_buffer.append(val)
                
                # Feedback visual en consola
                if len(data_buffer) % 500 == 0:
                    pct = (len(data_buffer) / SAMPLES_FOR_AI) * 100
                    sys.stdout.write(f"\rCargando IA: {int(pct)}% ")
                    sys.stdout.flush()

                # 3. Disparo de IA
                if len(data_buffer) >= SAMPLES_FOR_AI:
                    print(f"\n>>> ANALIZANDO...", end="")
                    
                    payload = {"data": data_buffer, "fs": 233}
                    try:
                        response = requests.post(f"{API_BASE}/predict", json=payload)
                        if response.status_code == 200:
                            res = response.json()
                            pred = res["prediction"]
                            conf = res["confidence"]
                            
                            c = Fore.RED if pred == 'Stress' else Fore.BLUE
                            print_status(f" RESULTADO: {pred} ({conf:.1%})", c)
                    except:
                        print(" Error API")
                    
                    data_buffer = [] # Reset buffer IA

            except ValueError:
                pass

    except KeyboardInterrupt:
        break