import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("🧪 Modelos para predecir cianobacteria")
st.caption("Tablero complementario para el trabajo de graduación titulado:")
st.caption("*Diseño de un sistema de detección de cianobacteria en cuerpos de agua por medio de aprendizaje automático.*")
st.caption("Realizado por: Paola Andrea Ayala Pineda.")

#-------------- Párrafo descriptivo del proyecto ---------------
st.set_page_config(layout="wide")
col_img, col_txt = st.columns([1, 2], gap="medium")

with col_img:
    st.image("imagenes/estanque1.jpeg", use_container_width=True)
    # st.caption("Estanque del jardín Botánico de la UVG")

with col_txt:
    texto = """
    Actualmente, la detección de cianobacterias en cuerpos de agua en lagos como lo son 
    el de de Amatitlán y Atitlán, se realiza por medio de un  análisis  de  laboratorio. 
    Sin embargo,  este  proceso  puede ser tardado , costoso  y  necesita  de  personal 
    capacitado para llevarlo a cabo. 
    
    Es por eso que, mediante el uso de métodos de aprendizaje automático, a continuación 
    se muestran  modelos  que  están entrenados con distintos cuerpos de agua los cuales 
    tienen  diferencias  en  su  calidad  de  agua  para tener un método de detección de 
    cianobacteria más amplio, económico y eficaz. 

    El cuerpo de agua que se utilizó para la validación de los modelos fue el estanque del 
    jardín botánico de la Universidad del Valle de Guatemala. 
    """
    st.markdown(
        f"""
        <div style="text-align: justify;">
            {texto}
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("Presiona un botón para elegir el modelo que se desee analizar.")

st.set_page_config(page_title="Selector de Modelo", page_icon="🧪")

#st.markdown("## Selecciona un modelo")

def go(page_stub: str):
    """
    Intenta cambiar de página usando varios patrones comunes.
    Ajusta si tus archivos tienen otro nombre o ubicación.
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
        go("pages/lago_amatitlan.py")

with c2:
    if st.button(
        "**Modelo 2**",
        help="Predice clorofila y ficocianina con datos de Atitlán.",
        use_container_width=True
    ):
        go("pages/lago_atitlan.py")

with c3:
    if st.button(
        "**Modelo 3**",
        help="Predice clorofila con datos de Amatitlán y Atitlán.",
        use_container_width=True
    ):
        go("pages/ambos_lagos.py")




