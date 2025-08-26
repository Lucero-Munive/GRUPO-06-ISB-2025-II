<!-- Primer entregable
Se utilizará HTML para hacer modificaciones estéticas del .md se cambian los encabezados en formato # a <h>-->
<div style="background-color: #c2d1f7ff; padding: 10px; border-radius: 0px;">
  <h1><b>Propuesta de Proyecto - Curso de ISB</b></h1>
</div>

<!-- Índice -->
<div style="background-color: #f4f7ffff; padding: 12px; border-radius: 0px; margin-bottom: 20px;">
  <b>Índice</b>
  <ul>
    <li><a href="#descripcion">Descripción</a></li>
    <li><a href="#opcion1">Opción 1: Bioimpedancia para la Detección Temprana...</a></li>
    <li><a href="#titulo-tentativo">Título tentativo</a></li>
    <li><a href="#problematica">Problemática a abordar</a></li>
    <li><a href="#objetivos">Objetivos a alcanzar</a></li>
    <li><a href="#herramientas">Herramientas a utilizar</a></li>
    <li><a href="#aporte">Aporte de los Integrantes</a></li>
  </ul>
</div>


*26/08/2025*

<h3 id="descripcion"><b>Descripción</b></h3>
<div style="text-align: justify;">
A continuación, se presenta la propuesta de proyecto para el curso de Introducción a las Señales Biomédicas (ISB).

Tal como se conversó en clase, esta propuesta detalla dos posibles temáticas para el desarrollo del proyecto final. La primera opción se centra en el análisis de señales de bioimpedancia eléctrica, una temática de especial interés debido a que se cuenta con un modelo de Machine Learning preexistente y funcional para la detección de colelitiasis, el cual ha sido replicado y mejorado a partir de un trabajo de investigación publicado.

No obstante, reconociendo que la viabilidad de la adquisición y el procesamiento de esta señal debe ser evaluada en el marco específico de los objetivos del curso, se presenta una segunda temática como una alternativa robusta y viable.

A continuación, se detalla la primera propuesta.
</div>

---
<br>
<div id="opcion1" style="background-color: #c2d1f7ff; padding: 10px; border-radius: 0px;">
  <h2><b>Opción 1:</b> Bioimpedancia para la Detección Temprana y Monitoreo de Patologías con Machine Learning y TinyML</h2>
</div>

<h3 id="titulo_1"><b>Título tentativo:</b></h3>
<div style="text-align: justify;">
Sistema no invasivo basado en bioimpedancia y machine-learning para la detección temprana y monitoreo de patologías(*)
</div>

<h3 id="problematica_1"><b>Problemática a abordar:</b></h3>
<div style="text-align: justify;">
Las enfermedades no transmisibles (ENT) constituyen la principal causa de morbimortalidad a nivel global, ejerciendo una presión creciente sobre la sostenibilidad de los sistemas de salud. En este contexto, un pilar fundamental para la gestión eficaz de estas patologías reside en el desarrollo de herramientas que permitan tanto una detección temprana como un monitoreo continuo de forma accesible, no invasiva y de bajo costo. Dicha necesidad se ve acentuada por las limitaciones inherentes a los métodos de diagnóstico convencionales, como las técnicas de imagenología o los análisis de laboratorio complejos. La frecuente centralización de estas tecnologías en centros hospitalarios restringe su aplicabilidad para el cribado masivo y el seguimiento ambulatorio, dificultando el acceso a diagnósticos oportunos para amplios sectores de la población, particularmente en entornos de atención primaria o con recursos limitados.

Ante esta problemática, el análisis de señales de bioimpedancia eléctrica (BIE) surge como una tecnología prometedora, siendo una técnica no invasiva, segura, portátil y de relativamente bajo costo en comparación a métodos de diagnóstico convencionales. Esta técnica se basa en la exposición a flujos de corriente eléctrica de tejidos biológicos, para caracterizar su conductividad eléctrica, ofreciendo información valiosa sobre la composición corporal, el estado de los fluidos y la integridad celular.
<div>

![imagen_1](https://www.researchgate.net/publication/319317929/figure/fig5/AS:534853667438592@1504530459710/Bioelectrical-Impedance-Analysis-BIA-principle-of-work.png)

<div align="center">
*Imágen 1: Bioelectrical Impedance Analysis (BIA) principle of work, tomada de: [ResearchGate](https://www.researchgate.net/figure/Bioelectrical-Impedance-Analysis-BIA-principle-of-work_fig5_319317929) <a href="#ref1">[ref]</a>* 
<div>


<div style="text-align: justify;">
La versatilidad y el potencial de la BIE puede ser aplicada a múltiples enfermedades no transmisibles, tanto para un diagnóstico temprano o pre-diagnóstico no invasivo, como para el monitoreo de síntomas que pueden alertar sobre la enfermedad. Entre estas podemos ejemplificar la detección temprana de colelitiasis (cálculos biliares), el monitoreo de insuficiencia cardiáca (IC), o incluso estimaciones de obesidad o sarcopenia.
<div>

<br>
<div style="text-align: justify;">
Dentro de los casos de aplicación de la BIE para el alcance del proyecto, se propone la integración en alguna de las dos primeras como:
<div>

<h4 id="calculos_1">1) <b>Detección Temprana de Colelitiasis:</b></h4>
<div style="text-align: justify; text-indent: 40px;">
Es una de las enfermedades digestivas más comunes, con alta prevalencia (6% de la población mundial), teniendo América del Sur la prevalencia combinada más alta, con un 11,2%. Es de especial interés el diagnóstico pre clínico o de etapas tempranas de la enfermedad, debido a su alta incidencia, así como a la poca sintomatología que presentan los pacientes en estadíos tempranos.

El diagnóstico estándar depende de la ecografía abdominal, una técnica no siempre disponible para un cribado masivo. Investigaciones recientes, como el estudio "Early prediction of gallstone disease with a machine learning-based method from bioimpedance and laboratory data", han revelado que las alteraciones metabólicas sistémicas que conducen a la formación de cálculos biliares modifican las propiedades eléctricas del cuerpo de una manera que la BIE puede detectar <a href="#ref1">[ref]</a>
</div>

<h4 id="ic_1">2) <b>Monitoreo de Insuficiencia cardíaca (IC):</b></h4>

<div style="text-align: justify; text-indent: 40px;">
Es un síndrome clínico complejo y progresivo, que afecta a más de 64 millones de personas en todo el mundo, y representa una de las principales causas de hospitalización en adultos mayores, suponiendo una tasa de 50% de mortalidad en los 5 primeros años desde el diagnóstico <a href="#ref1">[ref]</a>. En el Perú, esta condición constituye una de las principales causas de ingreso en servicios de cardiología en pacientes de la tercera edad, y representando hasta un 7.2% de mortalidad intrahospitalaria en hospitales referentes nacionales como el hospital Edgardo Rebagliati <a href="#ref1">[ref]</a>.

Uno de los principales desafíos en el manejo crónico de la IC es la prevención de episodios de descomposición aguda, los cuales son la principal causa de hospitalizaciones recurrentes. Estos eventos de descompensación son predecidos por una acumulación lenta y progresiva de fluidos (edemas o retención de líquidos), un estado que a menudo es subclínico en sus fases iniciales, sin embargo debido a la naturaleza del edema (acumulación hídrica), que supone cambios en la conductividad de los tejidos corporales, la utilización de un sensor de bioimpedancia para cuantificar la sobrecarga hídrica, como el propuesto en "Bioimpedance Spectroscopy-Based Edema Supervision Wearable System for Noninvasive Monitoring of Heart Failure" representan alternativas de especial interés. <a href="#ref2">[ref]</a>
</div>
<br>


<h3 id="objetivos_1"><b>Objetivos a alcanzar:</b></h3>

<ol>
  <li>
    <b>Objetivo General:</b>
    <ol type="a">
      <li>
        Desarrollar un sistema no invasivo basado en cuantificación de parámetros de bioimpedancia eléctrica y algoritmos de machine learning capaz de funcionar como una herramienta de apoyo para la detección temprana de colelitiasis o para el monitoreo de la acumulación de fluidos en pacientes diagnosticados con insuficiencia cardíaca. Integrando adquisición de señales, procesamiento, desarrollo del modelo de AI y despliegue en dispositivos físicos mediante TinyML.
      </li>
    </ol>
  </li>
  <br>
  <li>
    <b>Objetivos Específicos:</b>
    <ol>
      <li>Definir y estandarizar el procedimiento para la obtención de señales de bioimpedancia, utilizando instrumentación comercial o un sistema basado en módulos, para asegurar la captura de datos confiables y reproducibles.</li>
      <li>Realizar un análisis descriptivo y exploratorio de los datos para identificar y caracterizar los patrones y diferencias significativas en los parámetros de bioimpedancia (resistencia, reactancia, ángulo de fase) asociados a la patología escogida.</li>
      <li>Desarrollar algoritmos para el preprocesamiento de las señales crudas y aplicar técnicas de ingeniería de características (feature engineering) para construir un conjunto de variables predictivas óptimas a partir de la data de BIE.</li>
      <li>Desarrollar y optimizar modelos de machine learning específicos:
        <ol type="a">
          <li>Entrenar y validar un modelo de clasificación supervisada para la detección de colelitiasis.</li>
          <li>Entrenar y validar un modelo para la estimación del nivel de sobrecarga hídrica en pacientes con insuficiencia cardíaca.</li>
        </ol>
      </li>
      <li>Optimizar e implementar uno de los modelos desarrollados en una plataforma de hardware de bajo consumo (TinyML).</li>
      <li>Validar el desempeño del modelo final utilizando un conjunto de datos de prueba independiente y métricas cuantitativas estándar para cuantificar su potencial clínico.</li>
    </ol>
  </li>


<h3 id="herramientas_1"><b>Herramientas a utilizar:</b></h3>

<div style="text-align: justify;">
Para la consecución de los objetivos planteados, el proyecto empleará un conjunto de herramientas de hardware, software y una revisión de posibles datasets.
<div>


<ol>
  <li>
    <h4 <b>Hardware:</b></h4>
    <ol type="a">
      <li>
        <h5 <b>Hardware para Adquisición de Datos y Prototipado:</b>  </h5>
        La selección del hardware de adquisición es un paso crítico que depende directamente de la técnica de análisis de bioimpedancia requerida para la patología en estudio. Principalmente, existen tres categorías de análisis:
        <table>
        <tr>
            <th>Técnica</th>
            <th>Descripción</th>
        </tr>
        <tr>
            <td><b>Monofrecuencia (SF-BIA)</b></td>
            <td>Utiliza una única frecuencia de corriente (generalmente 50 kHz) para estimar la composición corporal total. Es la técnica más común en dispositivos comerciales de bajo costo.</td>
        </tr>
        <tr>
            <td><b>Multifrecuencia (MF-BIA)</b></td>
            <td>Emplea varias frecuencias discretas. Se basa en el principio de que las corrientes de baja frecuencia no penetran las membranas celulares y fluyen por el agua extracelular (ECW), mientras que las de alta frecuencia atraviesan las células, midiendo el agua corporal total. Esto permite diferenciar entre los compartimentos de fluido.</td>
        </tr>
        <tr>
            <td><b>Espectroscopia (BIS)</b></td>
            <td>Realiza un barrido a través de un amplio espectro de frecuencias para modelar la respuesta eléctrica completa de los tejidos. Es la técnica más precisa para la evaluación del estado de los fluidos.</td>
        </table>
      </li>
      <li>
        <h5 <b>Balanzas de Bioimpedancia Comerciales:</b>  </h5>
        Entre los productos comerciales, más utilizados para obtención de datos generales de bioimpedancia, podemos encontrar la Xiaomi Mi Body Composition Scale 2, la cuál es una alternativa prometedora dada su disponibilidad, bajo costo y la capacidad documentada para extraer sus datos de impedancia a través de librerías de código abierto vía Bluetooth. Adicionalmente, se podrán utilizar otros dispositivos como la Balanza Control Corporal Bioimpedancia Euromar o similares para realizar mediciones complementarias y aprovechar los electrodos colocados en las manos para un BIA segmental, así como la realización de medidas comparativas para asegurar la consistencia de los datos.
      </li>
      <li>
        <h5 <b>Módulos Analizadores de Impedancia:</b>  </h5>
        Se tiene en consideración el uso de circuitos integrados especializados AD5933 o el AD5941 de la línea multicanal AD594X. Estos componentes permitirían un análisis de espectroscopia de bioimpedancia completo (barrido en frecuencia), ofreciendo datos mucho más ricos. Sin embargo, debido al coste, la complejidad en su implementación y a su disponibilidad limitada en el mercado peruano, se consideran una vía secundaria o de expansión.
      </li>
      <li>
        <h5 <b>Microcontroladores:</b>  </h5>
        El modelo de machine learning final será optimizado e implementado en una plataforma de bajo consumo. Las opciones principales son el Arduino Nano 33 BLE Sense y el ESP32, elegidos por su capacidad de cómputo, conectividad inalámbrica y compatibilidad TinyML.
      </li>
    </ol>
  </li>
  <br>
  <li>
    <h4 <b>Software:</b></h4>
    El proyecto se apoyará en un ecosistema de software de código abierto que abarca desde la gestión de datos hasta el despliegue del modelo en el dispositivo final.
    <ol type="a">
      <li>
        <h5 <b>Almacenamiento y Gestión de Datos:</b>  </h5>
        Se implementará una estrategia de almacenamiento dual para garantizar la flexibilidad y seguridad de los datos. Inicialmente, los datos recolectados se almacenarán localmente en formatos estructurados como CSV para el análisis y prototipado offline. Para facilitar la recolección en tiempo real y la colaboración, se utilizará una base de datos en la nube con un plan gratuito robusto, siendo Google Firebase (Firestore) la opción principal.
      </li>
      <li>
        <h5 <b>Análisis, Procesamiento y Modelado:</b>  </h5>
        El desarrollo se centrará en el lenguaje Python dentro de entornos de trabajo interactivos como Jupyter Notebooks o Google Colab. Se prevé la utilización del siguiente conjunto de librerías científicas:
        <ul>
            <li> Pandas y NumPy: Para la manipulación, operaciones numéricas con los datasets.
            </li>
            <li> SciPy: Para el procesamiento de señales.
            </li>
            <li> Matplotlib y Seaborn: Para la visualización de datos.
            </li>
            <li> Scikit-learn: Para implementar, entrenar y evaluar los modelos de ML clásicos.
            </li>
            <li> TensorFlow: Para el diseño y entrenamiento de modelos de DL en caso de ser requeridos.
            </li>
        </ul>
      </li>
      <li>
        <h5 <b>Despliegue en Dispositivos Embebidos (TinyML):</b> </h5> 
        Uso de TensorFlow Lite for Microcontrollers para convertir, optimizar e implementar los modelos en hardware seleccionado.
      </li>
    </ol>
  </li>
  <br>
  <li>
    <h4 <b>Datasets:</b> </h4>
    <table>
  <tr>
    <th>Nombre del Dataset</th>
    <th>Resumen y Relevancia para el Proyecto</th>
    <th>Acceso</th>
  </tr>
  <tr>
    <td><b>Gallstone Prediction Dataset (UCI)</b></td>
    <td>Dataset Principal. Contiene los datos exactos del paper de referencia para colelitiasis (BIE, demografía, laboratorios).</td>
    <td><a href="https://archive.ics.uci.edu/dataset/1150/gallstone-1">https://archive.ics.uci.edu/dataset/1150/gallstone-1</a></td>
  </tr>
  <tr>
    <td><b>BIA and Echocardiography in Heart Disease</b></td>
    <td>Dataset pequeño (n=40), vincula directamente mediciones de BIE con parámetros ecocardiográficos en pacientes con cardiopatía.</td>
    <td><a href="https://figshare.com/articles/dataset/Echocardiographic_and_bioimpedance_findings_/5404246?file=9310174">https://figshare.com/articles/dataset/Echocardiographic_and_bioimpedance_findings_/5404246?file=9310174</a></td>
  </tr>
  <tr>
    <td><b>NHANES (Ciclos con BIE, ej. 1999-2004)</b></td>
    <td>El dataset público más grande con BIE. Miles de registros de la población general de EE.UU. con BIE (resistencia a 50 kHz) y datos de composición corporal por DEXA.</td>
    <td><a href="https://wwwn.cdc.gov/Nchs/Nhanes/Search/DataPage.aspx">https://wwwn.cdc.gov/Nchs/Nhanes/Search/DataPage.aspx</a></td>
  </tr>
  <tr>
    <td><b>Bioimpedance for Sarcopenia Screening</b></td>
    <td>Dataset de un estudio para detectar sarcopenia. Contiene datos de BIE con una etiqueta clínica clara.</td>
    <td><a href="https://zenodo.org/records/6378787">https://zenodo.org/records/6378787</a></td>
  </tr>
  <tr>
    <td><b>Reference Percentiles for BIA Phase Angle</b></td>
    <td>Tabla de referencia con valores normales del Ángulo de Fase, aplicable en el feature engineering.</td>
    <td><a href="https://figshare.com/articles/dataset/Table_1_Brazilian_Reference_Percentiles_for_Bioimpedance_Phase_Angle_of_Healthy_Individuals_XLSX/20240652?file=36174948">Figshare </td>
  </tr>
  <tr>
    <td><b>UK Biobank</b></td>
    <td>Dataset (~500k participantes). Contiene mediciones de BIE y datos clínicos. Requiere una aplicación de investigación formal.</td>
    <td>UK Biobank (Acceso Restringido)</td>
  </tr>
  <tr>
    <td><b>ESSE-RF Study (Russian BIA Reference Data)</b></td>
    <td>Publicación del estudio ESSE-RF con datos de referencia de BIE de más de 13,000 adultos en Rusia. Requiere una aplicación de investigación formal.</td>
    <td>PubMed (Publicación del Estudio) (Acceso a datos restringido)</td>
  </tr>
</table>
  </li>








## Referencias
<ol>
  <li id="ref1">
    Early prediction of gallstone disease with a machine learning-based method from bioimpedance and laboratory data. <a href="https://www.researchgate.net/publication/319317929">Enlace</a>
  </li>

## Aporte de los Integrantes

| Integrante               | Aporte   |
|--------------------------|----------|
| Alvaro Untiveros         | 33.33 %  |
| Lucero Munive            | 33.33 %  |
| Fiorella Pérez           | 33.33 %  |
