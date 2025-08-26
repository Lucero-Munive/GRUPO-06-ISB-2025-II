## Explicación de Códigos

En esta carpeta se comparte un avance inicial de código basado en el artículo:  
[Early prediction of gallstone disease with a machine learning model](https://journals.lww.com/md-journal/fulltext/2024/02230/early_prediction_of_gallstone_disease_with_a.40.aspx).

### Descripción general

- **Replicación de resultados:**  
    Se replican los resultados del artículo, obteniendo las mejores métricas para el modelo XGBoosting.
- **Optimización de hiperparámetros:**  
    Se utiliza Optuna para optimizar los hiperparámetros, logrando un mayor número de estimadores y un learning rate más bajo, lo que mejora las métricas principales de *accuracy* y *AUC*.
- **Interpretabilidad:**
    Se incluye la interpretabilidad del modelo, usando SHAP para entender qué variables (a parte del análisis de covarianza del ANOVA) son las que más influyeron en el entrenamiento del modelo para la toma de decisiones.

### Archivos incluidos

- **Primer archivo:**  
    Contiene el modelo inicial, usando las 32 variables seleccionadas por los autores mediante ANOVA.
- **Segundo archivo:**  
    Propone el uso de características basadas en bioimpedancia, eliminando manualmente los datos de análisis de laboratorio para evaluar el desempeño del modelo solo con este tipo de datos.

### Observaciones

Aunque se observa una reducción en la precisión al eliminar características de laboratorio (que presentaban correlaciones media-fuerte), la optimización de hiperparámetros permite mantener un *accuracy* superior al 70%. Esto sugiere que es pertinente evaluar el potencial de esta idea en datos del contexto peruano y considerar su posible despliegue en modelos de bajo consumo, como TinyML.
