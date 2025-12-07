#ifndef ECG_BPM_H
#define ECG_BPM_H

#include <Arduino.h>

class EcgBPM {
private:
    float threshold;
    unsigned long lastPeakTime;
    float bpm;
    bool peakDetected;
    
    // Para el cálculo de media móvil del umbral
    float signalMean;

public:
    void begin() {
        threshold = 100.0; // Umbral inicial (se auto-ajusta)
        lastPeakTime = 0;
        bpm = 0.0;
        peakDetected = false;
        signalMean = 0.0;
    }

    // Procesa la muestra y devuelve TRUE si encontró un pico AHORA
    bool update(float sample) {
        unsigned long now = millis();
        peakDetected = false;

        // 1. Valor absoluto de la señal filtrada (centrada en 0)
        float absVal = abs(sample);

        // 2. Promedio dinámico para el nivel de ruido
        signalMean = (0.99 * signalMean) + (0.01 * absVal);

        // 3. Umbral Dinámico: El pico debe ser significativamente mayor al promedio
        // Ajustamos el umbral para que sea el doble del ruido base
        float dynamicThreshold = signalMean * 4.0;
        if (dynamicThreshold < 50) dynamicThreshold = 50; // Mínimo de seguridad

        // 4. Detección
        // Condiciones:
        // A. La señal supera el umbral dinámico
        // B. Han pasado al menos 300ms desde el último pico (Periodo Refractario)
        //    (300ms limita a max 200 BPM, evita falsos dobles picos)
        if (absVal > dynamicThreshold && (now - lastPeakTime > 300)) {
            
            // Calculamos BPM instantáneo
            unsigned long delta = now - lastPeakTime;
            if (lastPeakTime != 0) { // Ignorar el primerísimo pico
                bpm = 60000.0 / delta;
            }
            
            lastPeakTime = now;
            peakDetected = true;
        }

        return peakDetected;
    }

    float getBPM() {
        return bpm;
    }
};

#endif