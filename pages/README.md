# 🤖 Modelos de Aprendizaje Automático (SVM · KNN · NN)

Este directorio contiene los scripts en **Python** utilizados para entrenar, evaluar y comparar diferentes modelos de **Machine Learning** aplicados a la estimación de **clorofila-a**, con el fin de detectar la presencia de **cianobacterias** en cuerpos de agua.

---

## 🧠 Modelos incluidos

- **SVM (Máquinas de Vectores de Soporte)** → Clasificación no lineal basada en hiperplanos óptimos.  
- **KNN (K-Vecinos más Cercanos)** → Clasificador basado en distancia entre muestras.  
- **NN (Red Neuronal Profunda)** → Modelo de regresión no lineal para predicción continua de clorofila-a.

---

## 📊 Funcionalidad principal

- Lectura y preprocesamiento de los **datasets institucionales (CEA y AMSA)**.  
- División de datos en entrenamiento y prueba.  
- Normalización y estandarización de variables.  
- Entrenamiento de los modelos y cálculo de métricas de rendimiento (**R²**, error medio, precisión, matrices de confusión).  
- Visualización de resultados para análisis comparativo.

---

## 🧩 Archivos principales

| Archivo | Descripción |
|----------|--------------|
| `svm_model.py` | Entrenamiento y validación del modelo SVM |
| `knn_model.py` | Implementación del clasificador KNN |
| `nn_model.py` | Entrenamiento de la red neuronal profunda |
| `lago_amatitlan.py` | Datos de AMSA |
| `lago_atitlan.py` | Datos de CEA |
| `ambos_lagos.py` | Integración de datos CEA + AMSA y pruebas combinadas |
---
