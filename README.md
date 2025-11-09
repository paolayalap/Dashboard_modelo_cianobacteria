<p align="center">
  <!-- Reemplaza las rutas por las de tus imágenes si ya las tienes en /imagenes -->
  <img src="imagenes/logo_UVG.jpg" alt="UVG" height="70">
  <img src="imagenes/ing_logo.png" alt="Facultad de Ingeniería" height="70">
</p>

<h1 align="center">
  Diseño de un sistema de detección de cianobacteria en cuerpos de agua por medio de aprendizaje automático.
</h1>

<h3 align="center">
  Trabajo de Graduación — Ingeniería Mecatrónica
</h3>

<p align="center">
  Autora:
  <strong>Paola Andrea Ayala Pineda</strong>
  <br>
  Asesor:
  <strong>Luis Alberto Rivera Estrada</strong>
  <br>
  Departamento de Ingeniería Electrónica, Mecatrónica y Biomédica — Universidad del Valle de Guatemala
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-en%20desarrollo-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/python-3.x-blue" alt="Python">
  <img src="https://img.shields.io/badge/streamlit-app-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/machine%20learning-active-purple" alt="ML">
  <img src="https://img.shields.io/badge/hardware-Arduino%20MEGA-orange" alt="Arduino">
</p>

---

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Dashboard_modelo_cianobacteria.streamlit.app/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

---


## 📘 Descripción general

Este repositorio reúne el código y los recursos del **trabajo de graduación de Ingeniería Mecatrónica (UVG)** enfocado en:

> Estimar la presencia de cianobacterias en cuerpos de agua mediante la **predicción de clorofila** usando parámetros físico-químicos y modelos de **aprendizaje automático**, integrados con un **sistema de sensores**.

El proyecto combina:

- ⚙️ **Hardware**: red de sensores conectados a un **Arduino MEGA 2560** para medir pH, temperatura, conductividad eléctrica, turbidez y oxígeno disuelto.
- 💻 **Aplicación web (Streamlit)**: interfaz interactiva para cargar datasets, entrenar modelos, visualizar resultados y aplicar el modelo a datos experimentales.

---

## 🌊 Contexto

Lagos como **Amatitlán** y **Atitlán** presentan proliferación recurrente de **cianobacterias**, afectando la calidad del agua, ecosistemas y salud humana.  
Este proyecto propone una herramienta:

- Accesible 🧪  
- Basada en datos reales de instituciones como **CEA** y **AMSA**  
- Capaz de apoyar decisiones de monitoreo y alerta temprana, sin depender exclusivamente de equipos de laboratorio costosos.

---

## 🧩 Estructura del repositorio

```bash
📂 .devcontainer       # Configuración de entorno en Codespaces / desarrollo
📂 .streamlit          # Configuración visual y parámetros de la app
📂 arduino             # Código para la red de sensores y adquisición de datos
📂 datasets_lagos      # Datasets institucionales (CEA, AMSA, combinados)
📂 imagenes            # Gráficas, figuras y recursos visuales
📂 pages               # Páginas internas de la aplicación Streamlit
📂 pruebas_piloto      # Datos del estanque experimental en UVG
📄 streamlit_app.py    # Archivo principal de la app en Streamlit
📄 requirements.txt    # Dependencias del proyecto
📄 runtime.txt         # Configuración para despliegue
📄 README.md           # Este archivo
´´´


# 🧠 ¿Qué hace la herramienta?

📥 **Carga datasets propios o incluidos en el repositorio.**

🧽 **Aplica limpieza, normalización y filtrado de datos.**

🤖 **Entrena y prueba distintos modelos de Machine Learning para estimar clorofila-a.**

---

## 📊 Muestra métricas como:

- Coeficiente de determinación (**R²**)  
- Error medio  
- **Matrices de confusión** y variantes con **lógica difusa**

---

## 🔁 Compara el desempeño entre:

- Datos de instituciones (**lagos reales**)  
- Datos experimentales del **estanque piloto**

---

## 🌐 Visualización intuitiva

Permite visualizar de forma sencilla si un conjunto de parámetros medidos sugiere **mayor o menor presencia de clorofila**, facilitando la interpretación de resultados tanto en datos históricos como en mediciones en tiempo real.


# 🔬 Resultados del modelo de red neuronal (CEA + AMSA)

---

## 📈 Resumen general

Se entrenaron modelos con datos de **CEA**, **AMSA** y su combinación.  
Los mejores resultados se obtuvieron con un **modelo de red neuronal profunda**, logrando:

- Excelente capacidad de **generalización**  
- Ajuste adecuado para estimar **clorofila-a** con parámetros físico-químicos medidos en campo  

Estos resultados demuestran el potencial del sistema para **monitorear la calidad del agua de manera económica y accesible**, integrando **inteligencia artificial y sensores de bajo costo**.

---

## 🧮 Ecuación de aproximación lineal

<p align="center">
  <img src="imagenes/EC3.jpg" alt="Ecuación de aproximación" width="500">
</p>

> Aproximación lineal de la red neuronal sobre datos CEA + AMSA.  
> Coeficiente de determinación: \( R^2 = 0.938 \)

---

## 📉 Curva de entrenamiento

<p align="center">
  <img src="imagenes/curvares.png" alt="Curva de entrenamiento" width="600">
</p>

**Descripción:**  
Durante las primeras épocas, la pérdida disminuye rápidamente indicando que el modelo aprende patrones iniciales.  
Posteriormente, la pérdida se estabiliza alrededor de un valor bajo, reflejando una buena generalización y ausencia de sobreajuste.

---

## 🧠 Matriz de confusión difusa

<p align="center">
  <img src="imagenes/matrizres.png" alt="Matriz de confusión difusa" width="600">
</p>

**Interpretación:**  
El modelo muestra alta precisión para las clases “Muy bajo” y “Muy alto”, con ligeras confusiones en los rangos intermedios, lo que indica una correcta discriminación de concentraciones extremas de clorofila-a.  
El uso de **lógica difusa** permite suavizar los límites entre categorías y representar mejor la naturaleza continua de las concentraciones.

---

## 🧭 Conclusión

Los resultados confirman que la red neuronal es una herramienta prometedora para la **estimación indirecta de cianobacterias** en cuerpos de agua, combinando datos medidos por sensores físicos con técnicas de aprendizaje automático.

---

📁 **Archivos visuales incluidos:**
- `imagenes/EC3.jpg` → Ecuación de aproximación lineal  
- `imagenes/curvares.png` → Curva de entrenamiento  
- `imagenes/matrizres.png` → Matriz de confusión con lógica difusa




