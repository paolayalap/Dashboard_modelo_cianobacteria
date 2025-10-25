# ============================================
# Página principal: Selector y portada
# ============================================
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración SIEMPRE al inicio (solo una vez)
st.set_page_config(page_title="Selector de Modelo", page_icon="🧪", layout="wide")

# ------------------ Encabezado ------------------
st.title("🧪 Modelos para predecir cianobacteria")
st.caption("Tablero complementario para el trabajo de graduación titulado:")
st.caption("*Diseño de un sistema de detección de cianobacteria en cuerpos de agua por medio de aprendizaje automático.*")
st.caption("Realizado por: Paola Andrea Ayala Pineda.")

# ------------------ Portada con imagen + texto ------------------
#col_img, col_txt = st.columns([1, 2], gap="medium")

#with col_img:
    # URL RAW correcta (no usar el enlace 'blob')
#    st.image(
#        "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/imagenes/estanque1.jpg",
 #       use_container_width=True
  #  )
    # st.caption("Estanque del Jardín Botánico de la UVG")

#with col_txt:
 #   texto = """
  #  Actualmente, la detección de cianobacterias en cuerpos de agua en lagos como 
   # Amatitlán y Atitlán se realiza por medio de un análisis de laboratorio. 
    #Sin embargo, este proceso puede ser tardado, costoso y requiere personal capacitado.

    #Por eso, mediante aprendizaje automático se entrenan modelos con distintos 
    #cuerpos de agua para lograr un método de detección más amplio, económico y eficaz.

    #El cuerpo de agua usado para validar los modelos fue el estanque del Jardín 
    #Botánico de la Universidad del Valle de Guatemala.
    """
    #st.markdown(f'<div style="text-align: justify;">{texto}</div>', unsafe_allow_html=True)

#st.write("Presiona un botón para elegir el modelo que desees analizar.")

# ------------------ Navegación entre páginas ------------------
def go(page_stub: str):
    """
    Intenta cambiar de página usando varios patrones comunes.
    Pasa solo el nombre base del archivo sin extensión, por ejemplo: 'lago_amatitlan'
    """
    # 1) /pages/<nombre>.py
    try:
        st.switch_page(f"pages/{page_stub}.py")
        return
    except Exception:
        pass

    # 2) nombre visible en el menú de páginas (sin .py)
    try:
        st.switch_page(page_stub)
        return
    except Exception:
        pass

    # 3) /<nombre>.py en raíz
    try:
        st.switch_page(f"{page_stub}.py")
        return
    except Exception:
        st.error(
            f"No pude abrir la página '{page_stub}'. "
            "Verifica el nombre y la ubicación del archivo (por ejemplo: 'pages/lago_amatitlan.py')."
        )

# --- Disposición horizontal de 3 botones ---
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    if st.button(
        "**Modelo 1**",
        help="Predice clorofila con datos de Amatitlán.",
        use_container_width=True
    ):
        go("lago_amatitlan")

with c2:
    if st.button(
        "**Modelo 2**",
        help="Predice clorofila y ficocianina con datos de Atitlán.",
        use_container_width=True
    ):
        go("lago_atitlan")

with c3:
    if st.button(
        "**Modelo 3**",
        help="Predice clorofila con datos de Amatitlán y Atitlán.",
        use_container_width=True
    ):
        go("ambos_lagos")


