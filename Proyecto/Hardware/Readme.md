# Arquitectura de Hardware - CardioCalm Wearable

Este directorio contiene el diseño electrónico, esquemáticos y el firmware del dispositivo wearable **CardioCalm**, encargado de la adquisición y transmisión de la señal ECG en tiempo real.

## Contenido

| Archivo / Carpeta | Descripción |
| :--- | :--- |
| `Lectura ECG/` | Código fuente del Firmware (PlatformIO / Arduino C++) |
| `Schematic_ISB_cardicalm_...pdf` | Diagrama esquemático del circuito electrónico |
| `Readme.md` | Este archivo de documentación |

---

## Especificaciones Técnicas

El hardware está basado en la arquitectura **ESP32-S3**, seleccionada por su doble núcleo (dual-core) y capacidades avanzadas de conectividad.

### Componentes Principales
1.  **Microcontrolador**: ESP32-S3-DevKitC-1.
    *   *Función*: Procesamiento central, ADC y transmisión BLE.
2.  **Sensor de ECG**: AD8232 (Single Lead Heart Rate Monitor).
    *   *Función*: Amplificación y filtrado analógico de la señal bioeléctrica.
    *   *Configuración*: Lead II (Brazo derecho - Pierna izquierda).
3.  **Pantalla**: OLED 0.96" I2C (SSD1306).
    *   *Función*: Biofeedback local (visualización de BPM y estado de conexión).
4.  **Alimentación**: Batería LiPo 3.7V con módulo de carga TP4056.

---

## Configuración de Pines (Pinout)

Esta configuración es **CRÍTICA** para el correcto funcionamiento del firmware. Se realizó un cambio importante respecto al diseño inicial para evitar conflictos con el WiFi.

| Componente | Pin del Componente | Pin ESP32 (GPIO) | Descripción Técnica |
| :--- | :--- | :--- | :--- |
| **AD8232 (Sensor)** | OUTPUT | **GPIO 4** | **ADC1_CH3**. Se usa ADC1 porque el ADC2 es bloqueado cuando el WiFi está activo. |
| | LO+ | GPIO 16 | Detección de electrodos desconectados (Leads Off). |
| | LO- | GPIO 17 | Detección de electrodos desconectados. |
| **OLED (Pantalla)** | SDA | GPIO 21 | Comunicación I2C (Data). |
| | SCL | GPIO 22 | Comunicación I2C (Clock). |
| | VCC | 3.3V | Alimentación lógica. |

---

## Firmware (`Lectura ECG/CardioCalm_Edge_S3`)

El firmware está desarrollado en **C++** utilizando el framework Arduino sobre PlatformIO.

### Lógica de Funcionamiento
1.  **Inicialización**: Configura el ADC1, el Bluetooth Low Energy (BLE) y la pantalla OLED.
2.  **Servidor BLE**:
    *   *Nombre del Dispositivo*: `CardioCalm-Wearable`
    *   *Service UUID*: `4fafc201-1fb5-459e-8fcc-c9c0f99421b1`
    *   *Characteristic UUID*: `beb5483e-36e1-4688-b7f5-ea07361b26a8` (Notificaciones habilitadas).
3.  **Loop Principal**:
    *   Lee el valor analógico del Pin 4.
    *   Aplica un filtro simple (promedio móvil) si es necesario.
    *   Calcula BPM localmente (algoritmo simple de detección de picos R-R) para mostrar en la OLED.
    *   **Transmisión**: Envía el dato crudo vía BLE al cliente (Frontend) cada ~3ms (aprox. 330Hz efectivos tras downsampling de transmisión).

### Instrucciones de Flasheo
1.  Instalar **VS Code** con la extensión **PlatformIO**.
2.  Abrir la carpeta `Lectura ECG/CardioCalm_Edge_S3`.
3.  Conectar el ESP32 vía USB.
4.  Ejecutar la tarea: `PlatformIO: Upload`.

---

## Diseño de Circuito

El diagrama esquemático (`Schematic_ISB_cardicalm...pdf`) detalla las conexiones físicas.
*   **Nota de Diseño**: Se ha priorizado la separación de las pistas analógicas (ECG) de las antenas de RF (BLE/WiFi) para minimizar la interferencia electromagnética.
*   **Electrodos**: Se recomienda el uso de electrodos de gel húmedo (Ag/AgCl) para obtener la mejor relación señal-ruido.
