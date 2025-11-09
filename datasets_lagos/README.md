# 💾 Datasets de entrenamiento — CEA y AMSA

Esta carpeta contiene los **archivos .csv utilizados para el entrenamiento y validación** de los modelos de aprendizaje automático (SVM, KNN y NN) implementados en la carpeta `pages/`.

Los datos provienen de **instituciones encargadas del monitoreo de la calidad del agua en Guatemala**, específicamente **CEA** y **AMSA**, y fueron procesados para ser compatibles con los algoritmos de predicción de **clorofila-a**.

---

## 📂 Archivos incluidos

| Archivo | Descripción |
|----------|-------------|
| `DATOS AMSA.csv` | Datos proporcionados por **AMSA (Autoridad para el Manejo Sustentable de la Cuenca del Lago de Amatitlán)**. Incluye mediciones de pH, temperatura, oxígeno disuelto, conductividad y turbidez. |
| `DATOS CEA.csv` | Datos recolectados por el **CEA (Centro de Estudios Ambientales de la UVG)** provenientes de cuerpos de agua como el Lago de Atitlán. |
| `DATOS CEA Y AMSA.csv` | Conjunto **fusionado** que combina ambos datasets (CEA + AMSA), utilizado para **entrenamientos más robustos y generalizables**. |

---

## 🎯 Propósito

Estos datasets permiten **entrenar, validar y comparar** distintos modelos de predicción de **clorofila-a**, estimando la presencia de **cianobacterias** con base en parámetros físico-químicos del agua.

---

## ⚙️ Notas

- Los archivos fueron **limpiados, normalizados y depurados** para asegurar coherencia entre unidades y formatos.  
- Cada dataset se puede seleccionar desde la interfaz principal de **Streamlit** para ejecutar el entrenamiento correspondiente.

---
