import asyncio
import struct
import aiohttp
import time
from collections import deque
from bleak import BleakClient
from colorama import init, Fore, Style

# --- CONFIGURACIÓN ---
# 1. Dirección MAC que acabamos de encontrar (¡NO LA BORRES!)
DEVICE_ADDRESS = "98:88:E0:14:CE:9D" 

# 2. UUIDs del ESP32 (Deben coincidir con tu código Arduino)
ECG_CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# 3. Tu API en la Nube (Google Cloud Run)
API_BASE = "https://cardiocalm-api-65187920779.us-central1.run.app"
# API_BASE = "http://127.0.0.1:8000" # Descomenta si quieres probar local primero

# 4. Parámetros de IA
SAMPLES_FOR_AI = 14000  # Ventana de 60s aprox
SEND_TO_STREAM_RATIO = 5 # Enviar 1 de cada 5 puntos a la web (para no saturar)

# --- ESTADO GLOBAL ---
data_buffer = []      # Acumula 60 segundos para la IA
stream_queue = deque() # Cola temporal para enviar a la web en vivo

init()

def print_status(msg, color=Fore.WHITE):
    print(f"{color}{msg}{Style.RESET_ALL}")

# --- FUNCIÓN: ENVIAR DATOS A LA API (WEB) ---
async def api_worker():
    """ Tarea en segundo plano que envía datos a la nube sin bloquear el Bluetooth """
    async with aiohttp.ClientSession() as session:
        while True:
            # A. ENVIAR STREAMING (VISUALIZACIÓN)
            if len(stream_queue) > 0:
                val = stream_queue.popleft()
                try:
                    # Enviamos dato rápido a /stream
                    async with session.post(f"{API_BASE}/stream", json={"val": val}) as resp:
                        pass # No necesitamos leer la respuesta para ganar velocidad
                except Exception as e:
                    print(f"Error stream: {e}")

            # B. ENVIAR BUFFER A IA (PREDICCIÓN)
            # Verificamos si tenemos suficientes datos en el buffer global
            # Nota: Para evitar condiciones de carrera, hacemos una copia si está lleno
            if len(data_buffer) >= SAMPLES_FOR_AI:
                print(f"\n{Fore.CYAN}>>> BUFFER LLENO ({len(data_buffer)}) - ENVIANDO A IA...{Style.RESET_ALL}")
                
                # Copiamos y limpiamos el buffer principal rápido
                payload_data = list(data_buffer)
                data_buffer.clear() 
                
                payload = {"data": payload_data, "fs": 233}
                
                try:
                    async with session.post(f"{API_BASE}/predict", json=payload) as response:
                        if response.status == 200:
                            res = await response.json()
                            pred = res.get("prediction", "Unknown")
                            conf = res.get("confidence", 0.0)
                            
                            c = Fore.RED if pred == 'Stress' else Fore.GREEN
                            print_status(f" RESULTADO NUBE: {pred} ({conf:.1%})", c)
                        else:
                            print_status(f" Error API: {response.status}", Fore.RED)
                except Exception as e:
                    print_status(f" Error enviando a IA: {e}", Fore.RED)

            # Pequeña pausa para no quemar la CPU
            await asyncio.sleep(0.01)

# --- CALLBACK: CUANDO LLEGA UN DATO DEL ESP32 ---
def notification_handler(sender, data):
    """ Se ejecuta automáticamente cada vez que el ESP32 manda un dato """
    global data_buffer
    
    # 1. Desempaquetar los 8 bytes (double) que manda el Arduino
    # '<d' significa: Little Endian (<), double (d)
    try:
        val = struct.unpack('<d', data)[0]
    except Exception as e:
        print(f"Error desempaquetando: {e}")
        return

    # 2. Agregar al Buffer de IA
    data_buffer.append(val)

    # 3. Agregar a la cola de Streaming (con diezmado)
    # Usamos un contador estático simulado o simple lógica de módulo
    if len(data_buffer) % SEND_TO_STREAM_RATIO == 0:
        stream_queue.append(val)

    # 4. Feedback Visual en consola (estilo barra de carga)
    if len(data_buffer) % 200 == 0:
        pct = (len(data_buffer) / SAMPLES_FOR_AI) * 100
        print(f"\rRecibiendo BLE: {val:.2f} | Buffer IA: {int(pct)}%", end="", flush=True)

# --- BUCLE PRINCIPAL ---
async def main():
    print_status(f"--- CARDIO CALM GATEWAY (BLE) ---", Fore.CYAN)
    print_status(f"Conectando a {DEVICE_ADDRESS}...", Fore.YELLOW)

    # Iniciar el worker de la API en paralelo
    asyncio.create_task(api_worker())

    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print_status(f"¡CONECTADO AL WEARABLE!", Fore.GREEN)
            
            # Suscribirse a las notificaciones (El ESP32 empieza a mandar datos)
            await client.start_notify(ECG_CHARACTERISTIC_UUID, notification_handler)
            
            print_status("Recibiendo datos y retransmitiendo a Google Cloud...", Fore.MAGENTA)
            
            # Mantenemos el programa corriendo indefinidamente
            while True:
                await asyncio.sleep(1) # El trabajo real ocurre en el callback y el worker
        else:
            print_status("No se pudo conectar.", Fore.RED)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDesconectando...")
    except Exception as e:
        print(f"\nError Fatal: {e}")