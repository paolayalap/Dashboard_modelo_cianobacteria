# 🧪 Pruebas piloto — Estanque UVG

Esta carpeta contiene los **dataframes recopilados durante las pruebas experimentales** realizadas en el estanque del Jardín Botánico de la Universidad del Valle de Guatemala (UVG).  
Los datos fueron obtenidos mediante la **boya multisensorial** con sensores de pH, temperatura, conductividad, oxígeno disuelto y turbidez.

---

## 📂 Archivos incluidos

| Archivo | Descripción |
|----------|-------------|
| `dataframe.csv` | Contiene **todos los datos recopilados** (aproximadamente 5000 registros) en el estanque durante las pruebas. Es el conjunto más completo y representa la base general de mediciones. |
| `dataframe1.csv` | Contiene **aproximadamente 120 datos seleccionados aleatoriamente** del conjunto principal (`dataframe.csv`). Se utilizó para generar la **predicción de la primera prueba piloto**. |
| `dataframe2.csv` | Contiene **todos los datos de la segunda prueba piloto**, con un total de **416 registros**, utilizados para validar el desempeño del modelo con nuevas mediciones. |

---

## 🎯 Propósito

Estos archivos se utilizan para **evaluar la generalización del modelo** entrenado con datos institucionales (CEA y AMSA) frente a datos reales obtenidos en condiciones controladas.  
Permiten analizar la **coherencia de las predicciones** y la **variabilidad del sistema** en campo.

---

## ⚙️ Notas

- Los valores fueron preprocesados para mantener el mismo formato de las variables utilizadas en los modelos principales.  
- Pueden ser utilizados directamente desde la aplicación Streamlit para obtener estimaciones de clorofila-a en tiempo real.

---
