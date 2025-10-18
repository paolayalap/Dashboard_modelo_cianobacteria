import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("🧪 Modelos para predecir cianobacteria")
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
        st.switch_page(f"Pages/{page_stub}.py")
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
        "Modelo 1",
        help="Predice clorofila con datos de Amatitlán.",
        use_container_width=True
    ):
        go("Pages/lago_amatitlan.py")

with c2:
    if st.button(
        "Modelo 2",
        help="Predice clorofila y ficocianina con datos de Atitlán.",
        use_container_width=True
    ):
        go("Pages/lago_atitlan.py")

with c3:
    if st.button(
        "Modelo 3",
        help="Predice clorofila y ficocianina con datos de Amatitlán y Atitlán.",
        use_container_width=True
    ):
        go("Pages/ambos_lagos.py")




