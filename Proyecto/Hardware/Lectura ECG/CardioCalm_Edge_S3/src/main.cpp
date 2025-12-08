#include <Arduino.h>
#include <Wire.h>
// ----------------- LIBRERÍAS DE PANTALLA -----------------
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
// ----------------- LIBRERÍAS DE COMUNICACIÓN INALÁMBRICA -----------------
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
// ----------------- LIBRERÍAS DE DSP Y HARDWARE -----------------
#include <MPU6050_light.h> 
#include "EcgFilters.h"
#include "EcgBPM.h"

// --- PINOUT (CAMBIO CRÍTICO) ---
// El Pin 5 es ADC2 y NO funciona con Bluetooth.
// Usamos Pin 36 (VP) que es ADC1 y sí funciona siempre.
const int PIN_ECG = 4; 
const int PIN_SDA = 8;
const int PIN_SCL = 9;

// --- CONFIGURACIÓN OLED ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// --- CONFIGURACIÓN BLE ---
#define SERVICE_UUID           "4fafc201-1fb5-459e-8fcc-c9c0f99421b1" 
#define ECG_CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8" 
BLEServer* pServer = NULL;
BLECharacteristic* pEcgCharacteristic = NULL;
bool deviceConnected = false; 

// --- OBJETOS ---
MPU6050 mpu(Wire);
EcgProcessor filtro;
EcgBPM detector;

// --- TIEMPO Y BUFFERS ---
const unsigned long SAMPLING_PERIOD_US = 1428; // 700Hz
unsigned long lastSampleTime = 0;
const int SCREEN_FPS = 20; 
unsigned long lastScreenUpdate = 0;

// Variables para BLE y DSP
const int SEND_SKIP = 3; // Downsampling a ~233 Hz
int sendCounter = 0; 
double cleanEcgValue = 0.0; 
int currentBPM = 0;
int currentRR = 0;

// Variables Gráficas OLED
const int GRAPH_W = 85; 
int ecgBuffer[GRAPH_W]; 
int buffIdx = 0;

// Clase Callbacks para eventos de conexión BLE
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("BLE: Cliente conectado.");
    }

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("BLE: Desconectado. Reiniciando publicidad...");
      BLEDevice::startAdvertising(); 
    }
};

void setup() {
    Serial.begin(921600); 
    Serial.println("--- CardioCalm AI: Final Version ---");
    
    // Configuración ADC para mayor estabilidad
    analogReadResolution(12);

    // 1. Inicializar I2C y Sensores
    Wire.begin(PIN_SDA, PIN_SCL);
    delay(100);

    // 2. Iniciar OLED
    if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { 
      Serial.println(F("Fallo OLED"));
    } else {
      display.clearDisplay();
      display.setTextColor(SSD1306_WHITE);
      display.setTextSize(1);
      display.setCursor(20, 20);
      display.println(F("CardioCalm AI"));
      display.setCursor(20, 35);
      display.println(F("Iniciando..."));
      display.display();
    }

    // 3. Inicializar MPU (IMU)
    byte status = mpu.begin();
    if(status != 0){
        Serial.println(F("Fallo MPU6050"));
    } else {
        Serial.println(F("Calibrando MPU... No mover."));
        mpu.calcOffsets(); 
    }

    // 4. Inicializar DSP
    filtro.begin();
    detector.begin();

    // Limpiar buffer gráfico
    for(int i=0; i<GRAPH_W; i++) ecgBuffer[i] = SCREEN_HEIGHT/2;

    // 5. Inicialización BLE 
    BLEDevice::init("CardioCalm-Wearable"); 
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);
    pEcgCharacteristic = pService->createCharacteristic(
                            ECG_CHARACTERISTIC_UUID,
                            BLECharacteristic::PROPERTY_READ |
                            BLECharacteristic::PROPERTY_NOTIFY 
                          );
    pEcgCharacteristic->addDescriptor(new BLE2902());
    pService->start();
    
    // Iniciar publicidad
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0); 
    BLEDevice::startAdvertising();
    
    Serial.println("BLE: Listo y visible.");
    display.clearDisplay();
}

void loop() {
    unsigned long currentTime = micros();

    // --- NÚCLEO MATEMÁTICO & DSP (700 Hz) ---
    if (currentTime - lastSampleTime >= SAMPLING_PERIOD_US) {
        lastSampleTime = currentTime;
        
        // 1. Adquisición (Mantengo tu lógica invertida del código bueno)
        double raw = 4095 - analogRead(PIN_ECG);
        
        // 2. Filtrado
        cleanEcgValue = filtro.apply(raw); 
        bool isPeak = detector.update(cleanEcgValue);

        // 3. Actualizar BPM si hay pico
        if (isPeak) {
            float bpmFloat = detector.getBPM();
            if (bpmFloat > 40 && bpmFloat < 200) {
                currentBPM = (int)bpmFloat;
                currentRR = 60000 / currentBPM;
            }
        }
        
        // 4. Llenar Buffer Gráfico (Diezmado x5)
        static int graphSkip = 0;
        graphSkip++;
        if (graphSkip >= 5) {
            graphSkip = 0;
            // Mapeo centrado en pantalla (ajusta -600/600 según ganancia de tu sensor)
            int y = map((int)cleanEcgValue, -600, 600, SCREEN_HEIGHT, 0); 
            
            // Clamping para evitar líneas locas fuera de pantalla
            if (y < 0) y = 0;
            if (y >= SCREEN_HEIGHT) y = SCREEN_HEIGHT-1;
            
            ecgBuffer[buffIdx] = y;
            buffIdx = (buffIdx + 1) % GRAPH_W;
        }

        // 5. ENVIAR POR BLE (Diezmado x3 -> ~233Hz)
        sendCounter++;
        if (sendCounter >= SEND_SKIP) {
            sendCounter = 0;
            
            if (deviceConnected) {
                mpu.update(); 
                
                // Envío de ECG Limpio (8 bytes double)
                // OJO: Python debe recibir esto como 'double' (struct.unpack)
                uint8_t buffer[8];
                memcpy(buffer, &cleanEcgValue, 8); 

                pEcgCharacteristic->setValue(buffer, 8); 
                pEcgCharacteristic->notify(); 
            }
        }
    }

    // --- NÚCLEO VISUAL (20 FPS) ---
    if (millis() - lastScreenUpdate > (1000/SCREEN_FPS)) {
        lastScreenUpdate = millis();
        display.clearDisplay();

        // A. DIBUJAR ONDA
        for (int i = 0; i < GRAPH_W - 1; i++) {
            int idx = (buffIdx + i) % GRAPH_W;
            int nextIdx = (buffIdx + i + 1) % GRAPH_W;
            display.drawLine(i, ecgBuffer[idx], i+1, ecgBuffer[nextIdx], SSD1306_WHITE);
        }

        // B. INTERFAZ DERECHA
        display.drawFastVLine(GRAPH_W, 0, SCREEN_HEIGHT, SSD1306_WHITE);
        
        // Estado BT
        display.setCursor(GRAPH_W + 5, 0);
        display.setTextSize(1);
        display.print(deviceConnected ? "BT:ON" : "BT:--");
        
        // BPM
        display.setCursor(GRAPH_W + 5, 12);
        display.println("BPM");
        display.setCursor(GRAPH_W + 5, 22);
        display.setTextSize(2);
        if(currentBPM > 0) display.println(currentBPM);
        else display.println("--");

        display.drawFastHLine(GRAPH_W, 40, SCREEN_WIDTH-GRAPH_W, SSD1306_WHITE);

        // R-R
        display.setCursor(GRAPH_W + 5, 45);
        display.setTextSize(1);
        display.println("R-R"); 
        display.setCursor(GRAPH_W + 5, 55);
        if(currentRR > 0) display.println(currentRR);
        else display.println("--");
        
        display.display();
    }
}