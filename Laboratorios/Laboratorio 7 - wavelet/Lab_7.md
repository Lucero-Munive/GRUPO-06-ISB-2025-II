# LAB 7 - Transformada de Wavelet
---

## 1. Introducción
El estudio de señales biomédicas ha dependido por mucho tiempo de herramientas clásicas como la Transformada de Fourier. Sin embargo, esta técnica no permite analizar bien fenómenos que cambian en el tiempo, ya que solo entrega información en el dominio de la frecuencia. Para intentar resolver esta limitación surgió la Transformada de Fourier de Tiempo Corto (STFT), pero aun así su capacidad de análisis es restringida porque usa ventanas fijas que obligan a sacrificar resolución en tiempo o en frecuencia. En este contexto, la Transformada Wavelet (WT) aparece como una alternativa más versátil, pues adapta la resolución según la frecuencia y permite observar detalles finos de la señal sin perder de vista su estructura general [1,2].

## ¿Qué hace la Transformada Wavelet?

La Transformada Wavelet (WT) permite realizar un análisis localizado en el tiempo de una señal completa, lo que significa que puede identificar eventos transitorios como picos breves, discontinuidades o rupturas que resultan complicados de detectar con otras herramientas de procesamiento.

Mientras que la Transformada de Fourier de Tiempo Corto (STFT) requiere definir una ventana fija, lo cual limita la resolución y puede ocultar ciertos detalles, el análisis wavelet ofrece una representación multiresolución[3].
<div align="center">
  <img src="Imagenes/FamiliaWavelet.png" alt="Ejemplos de wavelets y sus aplicaciones" width="700">
  <p><b>Figura 1.</b> Ejemplos de distintas familias de wavelets y sus aplicaciones.</p>
</div>

- a) **Análisis tiempo–frecuencia** mediante wavelets continuas como Morlet y Bump, que permiten estudiar señales no estacionarias en múltiples escalas. 
- b) **Detección de bordes y características** usando wavelets ortogonales (Haar, db4, db2), útiles para identificar cambios bruscos y transiciones en la señal. 
- c) **Denoising**, ejemplificado con la Symlet (sym4), empleada en la reducción de ruido en señales biomédicas como ECG y EEG.
- d) **Compresión** con wavelets biortogonales (bior4.4), que permiten representar y almacenar señales e imágenes con alta eficiencia manteniendo la calidad de reconstrucción.

## Avances en clasificación de señales cardíacas
En el área cardiaca, la WT se ha utilizado para diseñar métodos que clasifican automáticamente los latidos del corazón. Gracias a técnicas como el Wavelet Scattering Transform, se logra una mayor resistencia al ruido y una mejor capacidad de reconocer patrones entre diferentes pacientes, lo que incrementa la precisión de los diagnósticos [1]. Además, la combinación de wavelets con modelos de aprendizaje automático ha permitido desarrollar sistemas de adquisición de ECG que fusionan varias arquitecturas, aumentando la sensibilidad y la especificidad al momento de detectar anomalías cardíacas[5].

## Aplicaciones en neuroingeniería y epilepsia
Por otro lado, en el campo de la neuroingeniería, las wavelets se han convertido en una herramienta clave para el análisis de electroencefalogramas (EEG). En particular, se han aplicado en la detección automática de crisis epilépticas, facilitando tanto el trabajo clínico como el desarrollo de sistemas de monitoreo continuo. De esta manera, no solo se ahorra tiempo en el análisis, sino que también se aumenta la confiabilidad de los resultados, lo que resulta esencial en pacientes con epilepsia que requieren seguimiento constante [5,6].

## Perspectivas de integración con IA y filtrado avanzado
Finalmente, las investigaciones más recientes apuntan a la integración de la WT con técnicas de inteligencia artificial y aprendizaje profundo. Gracias a ello, se han diseñado bancos de filtros capaces de reconstruir señales discretas sin pérdidas, lo cual mejora notablemente el procesamiento en tiempo real [2,5]. Al mismo tiempo, estas aplicaciones ya se están implementando en sensores inteligentes y plataformas de telemedicina, lo que abre la puerta a una medicina más personalizada y a sistemas de monitoreo remoto de alta eficiencia [5].

# 2. Objetivos específicos
- Seleccionar y justificar la familia de wavelets adecuada para el procesamiento de señales biomédicas (ECG, EMG y EEG).
- Implementar experimentalmente el filtrado y reconstrucción de las señales ECG, EMG y EEG, aplicando la DWT.
- Evaluar y discutir la efectividad del filtrado mediante análisis comparativo, contrastando las señales antes y después del procesamiento para mejorar la calidad de la señal.
# 3. Diseño del filtro
## Elección de familia wavelet:
Se eligió la familia wavelet Daubechies 4 (db4) para el procesamiento de las señales de ECG,EMG y EEG , ya que ,ha demostrado ser una herramienta robusta y eficiente, especialmente en contextos de monitoreo ambulatorio y plataformas portátiles como el BITalino [7].Ademas,su estructura compacta permite preservar la morfología de las señales sin introducir distorsiones, lo cual es crucial en la detección de eventos fisiológicos relevantes como los picos R en ECG [8]. En estudios recientes, db4 ha sido comparada favorablemente frente a otras wavelets como Haar, Symlet y Coiflet, mostrando mejor sensibilidad para transitorios rápidos y mayor resolución temporal en señales no estacionarias como EMG y EEG [9].De forma complementaria, un análisis comparativo reciente sobre denoising en ECG confirma que db4 ofrece una relación señal-ruido superior y menor error cuadrático medio frente a otras wavelets, lo que respalda su elección en entornos clínicos y portátiles [10]. Por otro lado, su implementación computacional resulta eficiente para sistemas embebidos, lo que facilita su integración en protocolos de medicina personalizada [9].

## Parámetros definidos
- Familia Wavelet : Daubechies 4 (db4) 
- Tipo de transformada : Discrete Wavelet Transform 
- Número de niveles : 4

<div align="center">
  <img src="Imagenes/db4wavelet.png" alt="db4" width="700">
  <p><b>Figura 2.</b> Wavelet db4</p>
</div>

# Referencias
- [1] Z. Liu, G. Yao, Q. Zhang, J. Zhang, and X. Zeng, “Wavelet Scattering Transform for ECG Beat Classification,” Computational and Mathematical Methods in Medicine, vol. 2020, Article ID 3215681, 2020.Disponible en: https://onlinelibrary.wiley.com/doi/10.1155/2020/3215681
- [2] C. Ramos, L. De la Cruz, et al., “Hybrid AI–Wavelet Model for Biomedical Signal Classification,” in Proc. IEEE INTERCON, 2023.Disponible en:https://ieeexplore.ieee.org/document/10326046
- [3] J. A. Cortés, H. B. Cano Garzón, and J. A. Chaves O., “Del análisis de Fourier a las Wavelets - Transformada Continua Wavelet (CWT),” Scientia Et Technica, vol. XIII, no. 37, pp. 133–138, Dec. 2007.
- [4] S. Su, Z. Zhu, S. Wan, F. Sheng, T. Xiong, et al., “An ECG Signal Acquisition and Analysis System Based on Machine Learning with Model Fusion,” Sensors, vol. 23, no. 17, 7643, 2023.Disponible en: https://www.mdpi.com/1424-8220/23/17/7643
- [5] J. Sánchez-Ramírez, R. Rosales-Roldán, et al., “Comparison of Wavelet Families in EEG Signal Processing for Epileptic Seizure Detection,” Procedia Computer Science, vol. 138, pp. 65–72, 2018. Disponible en: https://www.sciencedirect.com/science/article/pii/S1877050918307865?via%3Dihub
- [6] O. Faust, U. R. Acharya, H. Adeli, and A. Adeli, “Wavelet-based EEG Processing for Computer-Aided Seizure Detection and Epilepsy Diagnosis,” Seizure, vol. 26, pp. 56–64, 2015.Disponible en:https://www.seizure-journal.com/article/S1059-1311(15)00013-8/fulltext
- [7] A. K. Singh, *Discrete Wavelet Transform: A Signal Processing Approach*, Wiley eBooks, IEEE Xplore, 2024. . Disponible en: https://ieeexplore.ieee.org/book/10518231
- [8] S. S. R. S. Rao et al., “An efficient Hilbert envelope and factor analysis based estimation of R-peaks in ECG signal,” *Analog Integrated Circuits and Signal Processing*, vol. 114, pp. 1–13, 2025. Disponible en: https://link.springer.com/article/10.1007/s10470-025-02508-8
- [9] A. S. R. R. R. Reddy et al., “Streamlined Signal Processing for Electrocardiogram ECG,” *Lecture Notes in Networks and Systems*, vol. 931, pp. 27–38, Springer, 2025. Disponible en: https://link.springer.com/chapter/10.1007/978-3-031-99339-8_3
- [10] A.M. Nelushi,CH. Manathunga,NGS. Shantha y T.Nakayama “Assessing Cardiac Sympatho-Vagal Balance Through Wavelet Transform Analysis of Heart Rate Variability,”  vol. 15, no. 4, Art. no. 1687, 2025. Disponible en: https://www.mdpi.com/2076-3417/15/4/1687






