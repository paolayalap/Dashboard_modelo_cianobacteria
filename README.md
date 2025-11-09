# 🌿 Herramienta de Predicción de Clorofila

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


