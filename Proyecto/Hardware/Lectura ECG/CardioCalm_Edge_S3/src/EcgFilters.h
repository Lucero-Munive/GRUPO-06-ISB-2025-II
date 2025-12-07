#ifndef ECG_FILTERS_H
#define ECG_FILTERS_H

#include <Arduino.h>

class Biquad {
public:
    double b0, b1, b2, a1, a2;
    double z1, z2;

    Biquad() {
        z1 = 0.0;
        z2 = 0.0;
    }

    // Calcula la salida del filtro muestra a muestra
    double step(double x) {
        double w = x - a1 * z1 - a2 * z2;
        double y = b0 * w + b1 * z1 + b2 * z2;
        z1 = w;
        z2 = z1; // z2 guarda el estado anterior
        // Corrección z-delay para estabilidad
        z2 = z1; 
        z1 = w;
        return y;
    }
    
    // Versión correcta form II transposed
    double process(double in) {
        double out = in * b0 + z1;
        z1 = in * b1 + z2 - out * a1;
        z2 = in * b2 - out * a2;
        return out;
    }
};

class EcgProcessor {
private:
    Biquad notchFilter;
    Biquad lowPass;
    Biquad highPass;

public:
    void begin() {
        // Coeficientes calculados para Sampling Rate = 700 Hz
        
        // 1. NOTCH a 60Hz (Elimina ruido eléctrico)
        // Q = 4.0
        notchFilter.b0 = 0.933267;
        notchFilter.b1 = -1.603639;
        notchFilter.b2 = 0.933267;
        notchFilter.a1 = -1.603639;
        notchFilter.a2 = 0.866534;

        // 2. LOW PASS a 40Hz (Elimina ruido muscular/EMG)
        // Butterworth 2nd order
        lowPass.b0 = 0.020083;
        lowPass.b1 = 0.040167;
        lowPass.b2 = 0.020083;
        lowPass.a1 = -1.561018;
        lowPass.a2 = 0.641352;

        // 3. HIGH PASS a 0.5Hz (Elimina deriva DC/Respiración)
        // Butterworth 2nd order
        highPass.b0 = 0.996787;
        highPass.b1 = -1.993574;
        highPass.b2 = 0.996787;
        highPass.a1 = -1.993563;
        highPass.a2 = 0.993585;
    }

    double apply(double input) {
        // Cascada de filtros: Entrada -> Notch -> LowPass -> HighPass -> Salida
        double step1 = notchFilter.process(input);
        double step2 = lowPass.process(step1);
        double step3 = highPass.process(step2);
        return step3;
    }
};

#endif