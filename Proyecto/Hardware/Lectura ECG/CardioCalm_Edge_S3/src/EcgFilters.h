#ifndef ECG_FILTERS_H
#define ECG_FILTERS_H

#include <Arduino.h>

// --- CLASE BIQUAD (MATEMÁTICA PURA) ---
class Biquad {
public:
    double b0, b1, b2, a1, a2;
    double z1, z2;

    Biquad() {
        z1 = 0.0;
        z2 = 0.0;
    }

    // Algoritmo "Direct Form II Transposed" (Estable)
    double process(double in) {
        double out = in * b0 + z1;
        z1 = in * b1 + z2 - out * a1;
        z2 = in * b2 - out * a2;
        return out;
    }
};

// --- CLASE SMOOTHER (NUEVO: PARA QUITAR LO BORROSO) ---
// Promedia los últimos N valores para eliminar ruido aleatorio
#define MA_SIZE 5 // Tamaño de ventana (5 muestras es ideal para 700Hz)

class MovingAverage {
private:
    double buffer[MA_SIZE];
    int index;
    double sum;
public:
    MovingAverage() {
        index = 0;
        sum = 0.0;
        for(int i=0; i<MA_SIZE; i++) buffer[i] = 0.0;
    }

    double process(double input) {
        sum -= buffer[index];       // Restar el valor más viejo
        buffer[index] = input;      // Guardar nuevo valor
        sum += input;               // Sumar nuevo valor
        index = (index + 1) % MA_SIZE;
        return sum / MA_SIZE;       // Devolver promedio
    }
};

// --- PROCESADOR PRINCIPAL ---
class EcgProcessor {
private:
    Biquad notchFilter;
    Biquad lowPass;
    Biquad highPass;
    MovingAverage smoother; // <--- AGREGADO

public:
    void begin() {
        // Coeficientes para Sampling Rate = 700 Hz
        
        // 1. NOTCH a 60Hz (Elimina ruido de enchufe)
        notchFilter.b0 = 0.933267;
        notchFilter.b1 = -1.603639;
        notchFilter.b2 = 0.933267;
        notchFilter.a1 = -1.603639;
        notchFilter.a2 = 0.866534;

        // 2. LOW PASS a 40Hz (Elimina ruido muscular)
        lowPass.b0 = 0.020083;
        lowPass.b1 = 0.040167;
        lowPass.b2 = 0.020083;
        lowPass.a1 = -1.561018;
        lowPass.a2 = 0.641352;

        // 3. HIGH PASS a 0.5Hz (Elimina deriva DC / respiración)
        highPass.b0 = 0.996787;
        highPass.b1 = -1.993574;
        highPass.b2 = 0.996787;
        highPass.a1 = -1.993563;
        highPass.a2 = 0.993585;
    }

    double apply(double input) {
        // CADENA DE PROCESAMIENTO:
        // Raw -> Notch -> LowPass -> HighPass -> SMOOTHER -> Salida
        
        double s1 = notchFilter.process(input);
        double s2 = lowPass.process(s1);
        double s3 = highPass.process(s2);
        
        // El paso final de "magia" para quitar lo borroso:
        return smoother.process(s3);
    }
};

#endif