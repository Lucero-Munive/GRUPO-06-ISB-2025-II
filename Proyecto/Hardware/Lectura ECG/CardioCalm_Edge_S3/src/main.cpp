#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_light.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "EcgFilters.h"
#include "EcgBPM.h"

// --- PINOUT ---
const int PIN_ECG = 5; // Lead II (La mejor señal)
const int PIN_SDA = 8;
const int PIN_SCL = 9;

// --- OLED CONFIG (0x3C) ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
// Reset pin -1 porque comparte reset con la placa
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// --- OBJETOS ---
MPU6050 mpu(Wire);
EcgProcessor filtro;
EcgBPM detector;

// --- TIEMPO ---
const unsigned long SAMPLING_PERIOD_US = 1428; // 700Hz
unsigned long lastSampleTime = 0;

// --- VARIABLES GRÁFICAS ---
const int GRAPH_W = 85; // Ancho zona gráfica (aprox 2/3 pantalla)
int ecgBuffer[GRAPH_W]; 
int buffIdx = 0;
unsigned long lastScreenUpdate = 0;
const int SCREEN_FPS = 20; // Refresco visual

// Variables para datos estables en pantalla
int currentBPM = 0;
int currentRR = 0;

void setup() {
  Serial.begin(921600); // Velocidad alta para Python
  
  // 1. Iniciar I2C primero
  Wire.begin(PIN_SDA, PIN_SCL);
  delay(100);

  // 2. Iniciar OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { 
    Serial.println(F("Fallo OLED - Revisar conexión"));
    // No detenemos el código, seguimos por si acaso es solo la pantalla
  } else {
    // Pantalla de carga
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(20, 20);
    display.println(F("CardioCalm AI"));
    display.setCursor(20, 35);
    display.println(F("Calibrando..."));
    display.display();
  }

  // 3. Iniciar Sensores y DSP
  byte status = mpu.begin();
  if(status == 0) {
    Serial.println(F("Calibrando IMU..."));
    mpu.calcOffsets();
  }
  
  filtro.begin();
  detector.begin();
  
  // Inicializar buffer gráfico en el centro
  for(int i=0; i<GRAPH_W; i++) ecgBuffer[i] = SCREEN_HEIGHT/2;
  
  delay(1000);
  display.clearDisplay();
}

void loop() {
  unsigned long currentTime = micros();

  // --- NÚCLEO MATEMÁTICO (700 Hz) ---
  if (currentTime - lastSampleTime >= SAMPLING_PERIOD_US) {
    lastSampleTime = currentTime;
    
    // 1. Adquisición (Invertido)
    double raw = 4095 - analogRead(PIN_ECG);
    
    // 2. Filtrado
    double clean = filtro.apply(raw);
    
    // 3. Detección
    bool isPeak = detector.update(clean);
    
    // 4. Actualizar datos si hay latido válido
    if (isPeak) {
        float bpmFloat = detector.getBPM();
        if (bpmFloat > 40 && bpmFloat < 200) {
            currentBPM = (int)bpmFloat;
            currentRR = 60000 / currentBPM;
        }
    }

    // 5. Llenar Buffer Gráfico (Diezmado x5 para que entre en pantalla)
    static int skip = 0;
    skip++;
    if (skip >= 5) {
        skip = 0;
        // Auto-escala visual: Ajusta -600 a 600 según la fuerza de tu señal
        int y = map((int)clean, -600, 600, SCREEN_HEIGHT, 0);
        
        // Limites de pantalla
        if (y < 0) y = 0;
        if (y >= SCREEN_HEIGHT) y = SCREEN_HEIGHT-1;
        
        ecgBuffer[buffIdx] = y;
        buffIdx = (buffIdx + 1) % GRAPH_W;
    }

    // 6. ENVIAR A PYTHON (Downsampling x3 -> ~233Hz)
    static int sendSkip = 0;
    sendSkip++;
    if (sendSkip >= 3) {
        sendSkip = 0;
        // Protocolo ligero: "E:valor,P:1/0"
        Serial.print("E:"); Serial.print(clean, 1); 
        Serial.print(",P:"); Serial.print(isPeak ? 1 : 0);
        
        mpu.update(); // Actualizar IMU
        Serial.print(",M:"); Serial.println(mpu.getAccX());
    }
  }

  // --- NÚCLEO VISUAL (~20 FPS) ---
  if (millis() - lastScreenUpdate > (1000/SCREEN_FPS)) {
    lastScreenUpdate = millis();
    display.clearDisplay();

    // A. DIBUJAR ONDA (Izquierda)
    for (int i = 0; i < GRAPH_W - 1; i++) {
        int idx = (buffIdx + i) % GRAPH_W;
        int nextIdx = (buffIdx + i + 1) % GRAPH_W;
        display.drawLine(i, ecgBuffer[idx], i+1, ecgBuffer[nextIdx], SSD1306_WHITE);
    }

    // B. DIBUJAR INTERFAZ (Derecha)
    // Línea separadora
    display.drawFastVLine(GRAPH_W, 0, SCREEN_HEIGHT, SSD1306_WHITE);
    
    // Bloque BPM
    display.setCursor(GRAPH_W + 5, 4);
    display.setTextSize(1);
    display.println("BPM");
    
    display.setCursor(GRAPH_W + 5, 16);
    if(currentBPM > 0) {
        display.setTextSize(2);
        display.println(currentBPM);
    } else {
        display.setTextSize(1);
        display.println("--");
    }

    // Separador horizontal
    display.drawFastHLine(GRAPH_W, 32, SCREEN_WIDTH-GRAPH_W, SSD1306_WHITE);

    // Bloque R-R
    display.setCursor(GRAPH_W + 5, 38);
    display.setTextSize(1);
    display.println("R-R");
    
    display.setCursor(GRAPH_W + 5, 50);
    if(currentRR > 0) {
        display.setTextSize(1);
        display.print(currentRR);
        // La unidad "ms" no cabe bien con texto grande, la omitimos por limpieza
    } else {
        display.println("--");
    }
    
    display.display();
  }
}