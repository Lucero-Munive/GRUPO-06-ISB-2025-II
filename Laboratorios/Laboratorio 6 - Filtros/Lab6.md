# **Laboratorio 6: Diseño y Aplicación de Filtros Digitales para Señales Biomédicas (EMG y ECG)**
---

## Índice
---
*   [Introducción](#introducción)
*   [Objetivos](#objetivos)
*   [Materiales](#materiales)
*   [Metodología](#metodología)
    *   [1. Carga y Conversión de Datos](#1-carga-y-conversión-de-datos)
    *   [2. Análisis Inicial de Señales Crudas](#2-análisis-inicial-de-señales-crudas)
    *   [3. Diseño y Carga de Filtros](#3-diseño-y-carga-de-filtros)
    *   [4. Aplicación y Evaluación de Filtros](#4-aplicación-y-evaluación-de-filtros)
*   [Resultados](#resultados)
    *   [1. Análisis de Señales Crudas](#1-análisis-de-señales-crudas)
    *   [2. Resultados del Filtrado para Señales EMG](#2-resultados-del-filtrado-para-señales-emg)
    *   [3. Resultados del Filtrado para Señales ECG](#3-resultados-del-filtrado-para-señales-ecg)
*   [Discusión](#discusión)
    *   [Análisis de los Resultados EMG](#análisis-de-los-resultados-emg)
    *   [Análisis de los Resultados ECG](#análisis-de-los-resultados-ecg)
*   [Conclusiones](#conclusiones)
*   [Referencias](#referencias)
*   [Aporte de cada integrante](#aporte-de-cada-integrante)

---

## Introducción
---
El procesamiento digital de señales biomédicas se ha consolidado como una herramienta esencial para mejorar la calidad de registros fisiológicos como la electromiografía (EMG) y la electrocardiografía (ECG). Estas señales, ampliamente utilizadas en el diagnóstico muscular y cardíaco, suelen estar contaminadas por ruido de línea, artefactos de movimiento o interferencias electromagnéticas, lo que dificulta su interpretación directa. Aplicar filtros digitales adecuados permite mejorar la relación señal/ruido (SNR) y preservar las características morfológicas relevantes para el análisis clínico[1,2].

### Ventajas de los filtros digitales frente a los analógicos
A diferencia de los filtros analógicos, los filtros digitales ofrecen ventajas como estabilidad, reproducibilidad y flexibilidad en su implementación. Mediante algoritmos computacionales, es posible ajustar parámetros con precisión, adaptarse a distintos tipos de señales y realizar análisis tanto en tiempo real como en post-procesamiento [3]. Además, plataformas como BITalino han facilitado la adquisición de señales biomédicas en entornos no controlados, lo que incrementa la necesidad de técnicas de filtrado robustas [4].

### Tipos de filtros digitales: FIR vs IIR
Los filtros digitales se dividen principalmente en dos categorías: FIR (Finite Impulse Response) e IIR (Infinite Impulse Response).
 - Los filtros FIR garantizan fase lineal y estabilidad absoluta, lo que los hace ideales para preservar la morfología de ondas ECG como P, QRS y T [1].
 - Los filtros IIR son más eficientes computacionalmente y permiten alcanzar respuestas similares a los filtros analógicos con menos coeficientes, aunque pueden introducir distorsión de fase si no se diseñan adecuadamente [3].

### Consideraciones de señales EMG y ECG
Cada tipo de señal posee un espectro de frecuencia característico. En el caso de la ECG, el rango útil suele estar entre 0.5 y 150 Hz, mientras que en la EMG se extiende de 10 a 500 Hz [1]. Por tanto, la elección del filtro debe considerar tanto la banda de interés como el tipo de ruido predominante.Por otro lado , investigaciones recientes han evaluado filtros como Butterworth, Chebyshev, Gaussian y convolucionales, no solo por su capacidad para atenuar interferencias, sino también por su impacto en el rendimiento de modelos de clasificación automática [3,5].

<div align="center">
  <img src="Imagenes/Filtros.png" alt="Diagrama de procesamiento digital de señales" width="600">
  <p><b>Figura 1.</b> Flujo de procesamiento digital de señales: la señal analógica pasa por un filtro antialiasing, luego se digitaliza mediante un ADC, es procesada digitalmente, convertida nuevamente a analógica con un DAC y finalmente suavizada con un filtro de reconstrucción.</p>
</div>

---

## Objetivos
---
- Aplicar diferentes filtros digitales a señales biomédicas de ECG y EMG adquiridas con BITalino, con el fin de reducir el ruido y resaltar las características relevantes de cada registro.
- Seleccionar y justificar el filtro más adecuado para cada caso (2 ECG y 2 EMG).

---

## Materiales
---

| Material | Foto | N° | Detalles |
|----------|------|:--:|----------|
| Kit BITalino (r)evolution | <p align="center"><img src="Imagenes/Bitalino.jpeg" alt="BITalino" width="350"/></p> | 1 | Placa de adquisición de bioseñales (ECG, EMG, EDA, etc.) con resolución ADC de 10 bits y frecuencia de muestreo configurable (hasta 1000 Hz). <br><br> **Incluye:** <br> • 1 cable de 2 hilos <br> • 1 cable de 3 hilos <br> • 5 electrodos <br> • 1 batería recargable LiPo 3.7 V <br> • 1 guía de inicio rápido <br> • 1 placa BITalino |
| Entorno de programación en Python | <p align="center"><img src="Imagenes/Python.png" alt="Python" width="350"/></p> | 1 | Lenguaje de programación utilizado para el procesamiento digital de señales.|
| Herramienta pyFDA | <p align="center"><img src="Imagenes/pyfda.png" alt="pyFDA" width="350"/></p> | 1 | Herramienta para el diseño, análisis y comparación de filtros digitales (FIR e IIR). Permite simular la respuesta en frecuencia y evaluar el desempeño de distintos filtros aplicados a señales biomédicas. |


En este cuaderno, se aborda el proceso de diseño, aplicación y evaluación de filtros digitales para el preprocesamiento de señales biomédicas reales. Se utilizarán dos tipos de señales adquiridas previamente con el dispositivo BITalino:

1.  **Electromiografía de Superficie (sEMG):** Registrada durante contracciones musculares del bíceps y tríceps.
2.  **Electrocardiografía (ECG):** Registrada en reposo y durante la recuperación post-ejercicio.

El objetivo principal es eliminar el ruido y los artefactos inherentes a cada señal para obtener una representación fidedigna de la actividad fisiológica. Para ello, se seguirá la siguiente metodología:

-   **Carga y Visualización de Datos Crudos:** Se cargarán las señales sin procesar y se convertirán a unidades físicas (mV) para establecer una línea base.
-   **Diseño de Filtros:** Se utilizará la herramienta `pyfdax` para diseñar un conjunto de filtros FIR e IIR con características específicas para cada tipo de señal.
-   **Aplicación y Comparación:** Cada filtro se aplicará a la señal correspondiente y se evaluará su rendimiento de forma visual (dominio del tiempo y frecuencia) y cuantitativa.
-   **Selección del Filtro Óptimo:** Se calculará la Relación Señal-Ruido (SNR) para cada caso y se generará una tabla comparativa para justificar la elección del filtro más adecuado para cada aplicación, priorizando tanto la limpieza de la señal como la preservación de su morfología característica.
---

## Metodología
---
El laboratorio se ejecutó siguiendo un flujo de trabajo sistemático en un entorno de Python, utilizando un Jupyter Notebook para el procesamiento y la visualización. La metodología se puede dividir en las siguientes etapas clave:

### 1. Carga y Conversión de Datos
El primer paso consistió en la carga de las cuatro señales crudas (`biceps_maximo.txt`, `triceps_maximo.txt`, `derivacion_2_reposo.txt`, `derivacion_2_actividad_fisica.txt`) desde sus archivos de texto. Se implementó una función para:
- Leer los valores numéricos de cada archivo.
- Convertir los datos crudos del Convertidor Analógico-Digital (ADC) a unidades físicas de milivoltios (mV), utilizando la fórmula proporcionada por el fabricante del BITalino y las ganancias específicas para los sensores de EMG (1009) y ECG (1100) [2, 6].
- Generar un vector de tiempo correspondiente para cada señal, basado en la frecuencia de muestreo de 1000 Hz.

### 2. Análisis Inicial de Señales Crudas
Antes de aplicar cualquier filtro, se realizó un análisis exploratorio de cada una de las cuatro señales para establecer una línea base y justificar la necesidad del filtrado. Para cada señal, se generó una visualización compuesta por:
- **Dominio del Tiempo:** Para observar la morfología general y la presencia de deriva en la línea base.
- **Espectro de Densidad de Potencia (PSD):** Para identificar las bandas de frecuencia donde se concentra la energía de la señal y del ruido.
- **Transformada Rápida de Fourier (FFT):** Para analizar la magnitud y fase de las componentes de frecuencia.

### 3. Diseño y Carga de Filtros
Se utilizó un conjunto de ocho filtros pre-diseñados (cuatro para EMG y cuatro para ECG), cuyos coeficientes fueron exportados previamente a archivos `.csv`. Estos filtros fueron diseñados para cumplir con las especificaciones teóricas de cada tipo de señal [1, 3, 4]:
- **Filtros EMG:** Filtros Pasa-Banda (FIR e IIR) para aislar la energía muscular entre ~20 Hz y ~450 Hz.
- **Filtros ECG:** Filtros Pasa-Bajo e IIR Rechaza-Banda (Notch) para atenuar el ruido de alta frecuencia y la interferencia de la red eléctrica, respectivamente.
#### 3.1. Diseño de Filtros EMG en `pyfdax`

Ahora que hemos justificado la necesidad de los filtros, procederemos a diseñar los **cuatro filtros de comparación para la señal EMG**.


##### Tabla Resumen de Filtros Diseñados

---

<div style="display: flex; align-items: flex-start; gap: 20px;">

<div style="flex: 1;">

###### 1️⃣ FIR_Equiripple_pasa_banda_EMG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Bandpass |
| **Tipo de Diseño** | FIR |
| **Método** | Equiripple |
| **Orden** | Minimum |
| **Frecuencias (Hz)** | f_SB=10, f_PB=20, f_PB2=450, f_SB2=460 |
| **Ripple (dB)** | A_SB=60, A_PB=1, A_SB2=60 |
| **β (Kaiser)** | — |

</div>

<div style="flex: 1;">
  <img src="Imagenes/Filtros/EMG/FIR_Equiripple_pasa_banda_EMG/magnitud.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/FIR_Equiripple_pasa_banda_EMG/fase.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/FIR_Equiripple_pasa_banda_EMG/polos_ceros.jpg" width="300"/>
</div>

</div>

---

<div style="display: flex; align-items: flex-start; gap: 20px;">

<div style="flex: 1;">

###### 2️⃣ FIR_Kaiser_pasa_banda_EMG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Bandpass |
| **Tipo de Diseño** | FIR |
| **Método** | Kaiser |
| **Orden** | Minimum |
| **Frecuencias (Hz)** | f_SB=10, f_PB=20, f_PB2=450, f_SB2=460 |
| **Ripple (dB)** | A_SB=60, A_PB=1, A_SB2=60 |
| **β (Kaiser)** | 5.65 |

</div>

<div style="flex: 1;">
  <img src="Imagenes/Filtros/EMG/FIR_Kaiser_pasa_banda_EMG/magnitud.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/FIR_Kaiser_pasa_banda_EMG/fase.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/FIR_Kaiser_pasa_banda_EMG/polos_ceros.jpg" width="300"/>
</div>

</div>

---

<div style="display: flex; align-items: flex-start; gap: 20px;">

<div style="flex: 1;">

###### 3️⃣ IIR_Butterworth_pasa_banda_EMG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Bandpass |
| **Tipo de Diseño** | IIR |
| **Método** | Butterworth |
| **Orden (N)** | 4 |
| **Frecuencias (Hz)** | f_PB=20, f_PB2=450 |
| **Ripple (dB)** | — |
| **β (Kaiser)** | — |

</div>

<div style="flex: 1;">
  <img src="Imagenes/Filtros/EMG/IIR_Butterworth_pasa_banda_EMG/magnitud.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/IIR_Butterworth_pasa_banda_EMG/fase.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/IIR_Butterworth_pasa_banda_EMG/polos_ceros.jpg" width="300"/>
</div>

</div>

---

<div style="display: flex; align-items: flex-start; gap: 20px;">

<div style="flex: 1;">

###### 4️⃣ IIR_Eliptico_pasa_banda_EMG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Bandpass |
| **Tipo de Diseño** | IIR |
| **Método** | Elliptic |
| **Orden (N)** | 4 |
| **Frecuencias (Hz)** | f_PB=20, f_PB2=450 |
| **Ripple (dB)** | A_SB=60, A_PB=1 |
| **β (Kaiser)** | — |

</div>

<div style="flex: 1;">
  <img src="Imagenes/Filtros/EMG/IIR_Eliptico_pasa_banda_EMG/magnitud.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/IIR_Eliptico_pasa_banda_EMG/fase.jpg" width="300"/><br>
  <img src="Imagenes/Filtros/EMG/IIR_Eliptico_pasa_banda_EMG/polos_ceros.jpg" width="300"/>
</div>

</div>

---

#### 3.2. Diseño de Filtros ECG en `pyfdax`

Para el análisis de las señales de ECG, se diseñó un conjunto de cuatro filtros con diferentes características para evaluar su capacidad de atenuar los artefactos preservando la morfología del latido. A continuación se detallan sus especificaciones y respuestas.

---

##### Tabla Resumen de Filtros Diseñados

---

<div style="display: flex; align-items/flex-start; gap: 20px;">
<div style="flex: 1;">

###### 1️⃣ FIR_Blackman_pasa_bajas_ECG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Lowpass |
| **Tipo de Diseño** | FIR |
| **Método** | Blackman |
| **Orden** | Minimum |
| **Frecuencias (Hz)** | f_PB=40, f_SB=50 |
| **Ripple (dB)** | A_PB=1, A_SB=80 |

</div>
<div style="flex: 1;">
  <img src="Imagenes/Filtros/ECG/FIR_Blackman_pasa_bajas_ECG/magnitud.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/FIR_Blackman_pasa_bajas_ECG/fase.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/FIR_Blackman_pasa_bajas_ECG/polos_ceros.png" width="300"/>
</div>
</div>

---

<div style="display: flex; align-items/flex-start; gap: 20px;">
<div style="flex: 1;">

###### 2️⃣ FIR_Hamming_pasa_bajas_ECG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Lowpass |
| **Tipo de Diseño** | FIR |
| **Método** | Hamming |
| **Orden** | Minimum |
| **Frecuencias (Hz)** | f_PB=40, f_SB=50 |
| **Ripple (dB)** | A_PB=1, A_SB=80 |

</div>
<div style="flex: 1;">
  <img src="Imagenes/Filtros/ECG/FIR_Hamming_pasa_bajas_ECG/magnitud.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/FIR_Hamming_pasa_bajas_ECG/fase.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/FIR_Hamming_pasa_bajas_ECG/polos_ceros.png" width="300"/>
</div>
</div>

---

<div style="display: flex; align-items/flex-start; gap: 20px;">
<div style="flex: 1;">

###### 3️⃣ IIR_Butter_rechaza_banda_ECG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Bandstop |
| **Tipo de Diseño** | IIR |
| **Método** | Butterworth |
| **Orden (N)** | 4 |
| **Frecuencias (Hz)** | f_SB=59, f_SB2=61 |
| **Ripple (dB)** | — |

</div>
<div style="flex: 1;">
  <img src="Imagenes/Filtros/ECG/IIR_Butter_rechaza_banda_ECG/magnitud.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/IIR_Butter_rechaza_banda_ECG/fase.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/IIR_Butter_rechaza_banda_ECG/polos_ceros.png" width="300"/>
</div>
</div>

---

<div style="display: flex; align-items/flex-start; gap: 20px;">
<div style="flex: 1;">

###### 4️⃣ IIR_Eliptico_pasa_baja_ECG.csv
| Parámetro | Valor |
|:-----------|:-------|
| **Tipo de Respuesta** | Lowpass |
| **Tipo de Diseño** | IIR |
| **Método** | Elliptic |
| **Orden (N)** | 4 |
| **Frecuencias (Hz)** | f_PB=40, f_SB=50 |
| **Ripple (dB)** | A_PB=1, A_SB=80 |

</div>
<div style="flex: 1;">
  <img src="Imagenes/Filtros/ECG/IIR_Eliptico_pasa_baja_ECG/magnitud.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/IIR_Eliptico_pasa_baja_ECG/fase.png" width="300"/><br>
  <img src="Imagenes/Filtros/ECG/IIR_Eliptico_pasa_baja_ECG/polos_ceros.png" width="300"/>
</div>
</div>

---


### 4. Aplicación y Evaluación de Filtros
Cada filtro se aplicó a su señal correspondiente utilizando un **filtrado de fase cero** para evitar la distorsión temporal. Se usaron las funciones `filtfilt` (para filtros FIR) y `sosfiltfilt` (para filtros IIR) de la librería SciPy, que garantizan la preservación de la alineación temporal de las características de la señal [5]. La evaluación se realizó de dos maneras:
- **Comparación Visual:** Se generaron gráficos superpuestos de la señal original vs. la señal filtrada para una evaluación cualitativa del rendimiento de cada filtro.
- **Evaluación Cuantitativa (SNR):** Se calculó la Relación Señal-Ruido para cada resultado. El método de cálculo se adaptó a la naturaleza de cada señal:
    - **Para EMG:** Se comparó la potencia de un segmento de contracción fuerte ("señal") con la de un segmento de reposo muscular ("ruido").
    - **Para ECG:** Se comparó la potencia alrededor de los picos R detectados (el complejo QRS, "señal") con la potencia del segmento isoeléctrico T-P ("ruido").

Este doble enfoque permitió una selección objetiva y justificada del filtro más adecuado para cada aplicación.


## Resultados
---
En esta sección se presentan los resultados obtenidos en cada etapa del procesamiento, desde la visualización inicial de las señales crudas hasta la evaluación cuantitativa de los filtros aplicados.

### 1. Análisis de Señales Crudas
Las siguientes figuras muestran las cuatro señales adquiridas directamente del dispositivo BITalino (convertidas a mV), antes de cualquier procesamiento. Se presentan en el dominio del tiempo y de la frecuencia (PSD y FFT) para caracterizar su estado inicial.

**Señales EMG:**
<div align="center">
  <img src="Lab6_filtros_files/Lab6_filtros_8_0.png" alt="Señal Cruda EMG Biceps" width="900">
  <p><b>Figura 2.</b> Visualización de la señal cruda para <code>EMG_Biceps</code>.</p>
  <img src="Lab6_filtros_files/Lab6_filtros_8_1.png" alt="Señal Cruda EMG Triceps" width="900">
  <p><b>Figura 3.</b> Visualización de la señal cruda para <code>EMG_Triceps</code>.</p>
</div>

**Señales ECG:**
<div align="center">
  <img src="Lab6_filtros_files/Lab6_filtros_8_2.png" alt="Señal Cruda ECG Reposo" width="900">
  <p><b>Figura 4.</b> Visualización de la señal cruda para <code>ECG_Reposo</code> (Derivación II).</p>
  <img src="Lab6_filtros_files/Lab6_filtros_8_3.png" alt="Señal Cruda ECG Post-Ejercicio" width="900">
  <p><b>Figura 5.</b> Visualización de la señal cruda para <code>ECG_PostEjercicio</code> (Derivación II).</p>
</div>

**Observación Inicial:** El análisis de la PSD en todas las señales revela una alta concentración de energía en bajas frecuencias (<10 Hz), característica de la deriva de línea base y artefactos de movimiento.

---

### 2. Resultados del Filtrado para Señales EMG

#### 2.1. Validación de los Filtros EMG
Se cargaron y visualizaron las respuestas en frecuencia de los cuatro filtros pasa-banda diseñados para EMG. Las gráficas confirman que los diseños cumplen con las especificaciones, mostrando las bandas de paso y atenuación esperadas.

<div align="center">
  <img src="Lab6_filtros_files/Lab6_filtros_13_1.png" alt="Filtro FIR Equiripple" width="700">
  <p><b>Figura 6.</b> Respuesta en frecuencia del filtro FIR Equiripple Pasa-Banda.</p>
  <img src="Lab6_filtros_files/Lab6_filtros_13_5.png" alt="Filtro IIR Butterworth" width="700">
  <p><b>Figura 7.</b> Respuesta en frecuencia del filtro IIR Butterworth Pasa-Banda.</p>
</div>

#### 2.2. Comparación Visual de la Aplicación
Los filtros se aplicaron a las señales de bíceps y tríceps. Las siguientes figuras muestran el resultado en una cuadrícula de 2x2 para cada señal.

<div align="center">
  <img src="Lab6_filtros_files/Lab6_filtros_15_1.png" alt="Comparación de filtros en Bíceps" width="900">
  <p><b>Figura 8.</b> Comparación del efecto de los cuatro filtros en la señal <code>EMG_Biceps</code>.</p>
  <img src="Lab6_filtros_files/Lab6_filtros_15_3.png" alt="Comparación de filtros en Tríceps" width="900">
  <p><b>Figura 9.</b> Comparación del efecto de los cuatro filtros en la señal <code>EMG_Triceps</code>.</p>
</div>

#### 2.3. Evaluación Cuantitativa (SNR) para EMG
Se calculó el SNR para cada señal (cruda y filtrada). Los resultados se resumen en la siguiente tabla:

| Señal | SNR Cruda | SNR FIR Equiripple | SNR FIR Kaiser | SNR IIR Butterworth | SNR IIR Eliptico |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EMG_Biceps** | 26.79 dB | 26.81 dB | 26.87 dB | **26.98 dB** | 27.17 dB |
| **EMG_Triceps**| 21.57 dB | 21.48 dB | 21.62 dB | **21.65 dB** | 21.42 dB |
---

*Tabla 1: Resultados de SNR para las señales EMG. Se resalta el rendimiento del filtro IIR Butterworth.*

---

### 3. Resultados del Filtrado para Señales ECG

#### 3.1. Validación de los Filtros ECG
Se visualizaron las respuestas en frecuencia de los cuatro filtros diseñados para ECG, confirmando sus características de pasa-bajo y rechaza-banda


<div align="center">
  <img src="Lab6_filtros_files/Lab6_filtros_22_1.png" alt="Filtro FIR Blackman" width="700">
  <p><b>Figura 10.</b> Respuesta en frecuencia del filtro FIR Blackman Pasa-Bajo.</p>
  <img src="Lab6_filtros_files/Lab6_filtros_22_5.png" alt="Filtro IIR Notch" width="700">
  <p><b>Figura 11.</b> Respuesta en frecuencia del filtro IIR Butterworth Rechaza-Banda (Notch).</p>
</div>

#### 3.2. Comparación Visual de la Aplicación
Los filtros se aplicaron a las señales de reposo y post-ejercicio, centrándose en un segmento corto para evaluar el impacto en la morfología del latido.

<div align="center">
  <img src="Lab6_filtros_files/Lab6_filtros_24_1.png" alt="Comparación de filtros en ECG Reposo" width="900">
  <p><b>Figura 12.</b> Comparación del efecto de los cuatro filtros en la señal <code>ECG_Reposo</code>.</p>
  <img src="Lab6_filtros_files/Lab6_filtros_24_3.png" alt="Comparación de filtros en ECG Post-Ejercicio" width="900">
  <p><b>Figura 13.</b> Comparación del efecto de los cuatro filtros en la señal <code>ECG_PostEjercicio</code>.</p>
</div>

#### 3.3. Evaluación Cuantitativa (SNR) para ECG
Se calculó el SNR para las señales de ECG utilizando el método de detección de picos R.

| Señal | SNR Cruda | SNR FIR Blackman | SNR FIR Hamming | SNR IIR Notch | SNR IIR Eliptico |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ECG_Reposo** | 11.54 dB | 13.66 dB | 13.82 dB | 13.68 dB | **14.24 dB** |
| **ECG_PostEjercicio** | 5.67 dB | 5.82 dB | 5.84 dB | 5.65 dB | **6.14 dB** |
---
*Tabla 2: Resultados de SNR para las señales ECG. Se resalta el rendimiento del filtro IIR Elíptico.*

---

## Discusión
---
Los resultados obtenidos permiten una discusión detallada sobre la efectividad de los diferentes enfoques de filtrado para cada tipo de señal biomédica.

### Análisis de los Resultados EMG
La evaluación de los filtros para las señales EMG arrojó dos hallazgos principales. Primero, la señal `EMG_Biceps` presentaba una calidad inicial muy alta (SNR > 26 dB), por lo que las mejoras obtenidas por los filtros fueron marginales. Esto demuestra que, si bien el filtrado siempre es una buena práctica, su impacto visible es menor en grabaciones de alta calidad. Segundo, para la señal `EMG_Triceps`, el filtro **IIR Butterworth** y el **FIR Kaiser** mostraron el mejor rendimiento, mejorando el SNR. En contraste, los filtros Equiripple y Elíptico, a pesar de tener transiciones más abruptas, redujeron ligeramente el SNR, probablemente debido al rizado que introducen en la banda de paso.

Considerando el rendimiento consistente en ambas señales y su eficiencia computacional (un IIR de orden 4 vs. un FIR de orden > 200), el **filtro IIR Butterworth** se perfila como la opción más robusta y equilibrada para el preprocesamiento general de señales EMG.

### Análisis de los Resultados ECG
Para las señales de ECG, la necesidad de filtrado fue mucho más evidente. La señal `ECG_PostEjercicio` cruda tenía un SNR de solo 5.67 dB, un valor bajo pero fisiológicamente esperado debido a los artefactos de movimiento y al ruido muscular (EMG) tras la actividad física.

En ambos casos (`ECG_Reposo` y `ECG_PostEjercicio`), el **filtro IIR Elíptico Pasa-Bajo** fue el claro ganador, proporcionando el mayor aumento de SNR. Esto se debe a su alta selectividad y su abrupta atenuación, que le permite eliminar eficazmente el ruido de alta frecuencia sin distorsionar significativamente el complejo QRS. El bajo rendimiento del filtro Notch confirma que la interferencia de 60 Hz no era la principal fuente de ruido, sino más bien el ruido muscular de banda ancha. La notable mejora del SNR en la señal post-ejercicio (de 5.67 dB a 6.14 dB) valida la capacidad de este filtro para recuperar información útil incluso en condiciones muy ruidosas.

---
## Conclusiones
---
Este laboratorio demostró con éxito el proceso completo de aplicación, comparación y selección de filtros digitales para dos tipos de señales biomédicas fundamentales, EMG y ECG. Se extraen las siguientes conclusiones principales:

1.  **La selección del filtro depende críticamente del tipo de señal:** Se confirmó que las señales EMG requieren un filtro Pasa-Banda con frecuencias de corte altas (e.g., 20-450 Hz), mientras que las señales ECG se benefician de un filtro Pasa-Bajo o Pasa-Banda con frecuencias de corte mucho más bajas (e.g., < 50 Hz) para eliminar la deriva de línea base y el ruido muscular.

2.  **El filtro IIR Butterworth Pasa-Banda fue la mejor opción para EMG**, ofreciendo un equilibrio óptimo entre la mejora del SNR, la consistencia entre diferentes señales y una alta eficiencia computacional, gracias a su banda de paso máximamente plana y su bajo orden.

3.  **El filtro IIR Elíptico Pasa-Bajo fue superior para el procesamiento de ECG**, demostrando la mayor capacidad para maximizar el SNR tanto en señales limpias como en señales altamente contaminadas por ruido, gracias a su excelente selectividad.

4.  **La evaluación cuantitativa (SNR) es indispensable.** La inspección visual por sí sola no fue suficiente para determinar el mejor filtro, especialmente en el caso de la EMG. El cálculo del SNR proporcionó una métrica objetiva y decisiva para justificar la selección del filtro más adecuado en cada caso.

---

## Referencias
---
[1] A. Chugh and C. Jain, “A Systematic Review on ECG and EMG Biomedical Signal Using Deep-Learning Approaches,” in Advances in Computational Intelligence and Communication: Proceedings of ICACIC 2023, Springer, 2023. Available: https://link.springer.com/chapter/10.1007/978-3-031-41925-6_11

[2] L. Sörnmo and P. Laguna, Bioelectrical Signal Processing in Cardiac and Neurological Applications. Burlington, MA, USA: Elsevier Academic Press, 2005. Available:https://www.elsevier.com/books/bioelectrical-signal-processing-in-cardiac-and-neurological-applications/sornmo/978-0-12-437552-9

[3] F. Zhou and D. Fang, “Multimodal ECG heartbeat classification method based on a convolutional neural network embedded with FCA,” Sci. Rep., vol. 14, no. 59311, 2024. Available: https://www.nature.com/articles/s41598-024-59311-0.pdf

[4] H. Silva et al., “BITalino: A Novel Hardware Framework for Physiological Computing,” in Proc. 36th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2014, pp. 593–596. Available: https://www.researchgate.net/publication/260987402_BITalino_A_Novel_Hardware_Framework_for_Physiological_Computing

[5] A. Sameh et al., “Digital phenotypes and digital biomarkers for health and diseases: a systematic review,” Artif. Intell. Rev., 2024. Available: https://link.springer.com/article/10.1007/s10462-024-11009-5

[6] De Luca CJ. Surface Electromyography: Detection and Recording. Delsys Inc; 2002.


## Aporte de cada integrante
| Integrante               | Aporte   |
|--------------------------|----------|
| Alvaro Untiveros         | 33.33 %  |
| Lucero Munive            | 33.33 %  |
| Fiorella Pérez           | 33.33 %  |