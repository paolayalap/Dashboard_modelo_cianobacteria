# ============================================
# Página principal: Selector de Modelo (MWE)
# ============================================
import streamlit as st

# Configuración: SOLO una vez y al inicio
st.set_page_config(page_title="Selector de Modelo", page_icon="🧪", layout="wide")

st.title("🧪 Modelos para predecir cianobacteria")
st.caption("Tablero complementario del TG de Paola Andrea Ayala Pineda.")
st.write("Presiona un botón para elegir el modelo que desees analizar.")

# --------- Navegación robusta ----------
def go(page_stub: str):
    """
    Cambia de página probando rutas típicas.
    Pasa el nombre base SIN .py y SIN 'pages/', por ejemplo: 'lago_amatitlan'
    """
    # 1) /pages/<nombre>.py
    try:
        st.switch_page(f"pages/{page_stub}.py"); return
    except Exception:
        pass
    # 2) nombre visible en el menú (sin .py)
    try:
        st.switch_page(page_stub); return
    except Exception:
        pass
    # 3) /<nombre>.py en raíz
    try:
        st.switch_page(f"{page_stub}.py"); return
    except Exception:
        st.error(
            f"No pude abrir la página '{page_stub}'. "
            "Asegúrate de que exista 'pages/" + page_stub + ".py' y que el nombre coincida."
        )

# --------- Botones en columnas ----------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    if st.button("Modelo 1", use_container_width=True):
        go("lago_amatitlan")    # <— stub correcto

with c2:
    if st.button("Modelo 2", use_container_width=True):
        go("lago_atitlan")      # <— stub correcto

with c3:
    if st.button("Modelo 3", use_container_width=True):
        go("ambos_lagos")       # <— stub correcto
