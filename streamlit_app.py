# ============================================
# Página principal: Selector de Modelo (MWE)
# ============================================
from pathlib import Path
import streamlit as st

# Configuración: SOLO una vez y al inicio
st.set_page_config(page_title="Selector de Modelo", page_icon="🧪", layout="wide")

st.title("🧪 Modelos para predecir cianobacteria")
st.caption("Tablero complementario del Trabajo de Graduación de Paola Andrea Ayala Pineda.")

left, right = st.columns([1, 2], gap="medium")

with left:
    st.markdown(
        """
        <div style="text-align: justify;">
            <p>
            Actualmente, la detección de cianobacterias en lagos como Amatitlán y Atitlán 
            depende de análisis de laboratorio. Aunque son confiables, requieren tiempo, 
            recursos y personal especializado.
            </p>
            <p>
            Este tablero presenta modelos de aprendizaje automático entrenados con datos 
            de distintos cuerpos de agua para ofrecer una detección más oportuna, 
            económica y escalable.
            </p>
            <p>
            Debido a que la cianobacteria se compone de distintos pigmentos, la clorofila
            en estos modelos, fue el indicador principal para la detección.
            Los parámetros utilizados para los modelos fueron: temperatura, conductividad, 
            oxígeno disuelto, pH y turbidez. 
            </p>
            <p>
            La validación de los modelos se realizó en el estanque del Jardín Botánico 
            de la Universidad del Valle de Guatemala.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Imagen en la columna derecha ----------
def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.is_file():
            return str(p)
    return None

# Carpeta esperada: tu_app/imagenes/estanque.png (misma raíz que este archivo)
BASE_DIR = Path(__file__).parent  # carpeta donde está este .py
CANDIDATE_PATHS = [
    "imagenes/estanque.png",                          # relativo a la raíz de la app
    BASE_DIR / "imagenes" / "estanque.png",           # relativo al archivo actual
    "static/imagenes/estanque.png",                   # si usas /static
    "assets/estanque.png",                            # alternativa común
]

img_path = first_existing(CANDIDATE_PATHS)

with right:
    if img_path:
        st.image(img_path, caption="Estanque del Jardín Botánico del Departamento de Biología de la Universidad del Valle de Guatemala", use_container_width=True)
        st.caption(f"📷 Cargada desde: `{img_path}`")
    else:
        st.warning(
            "No encontré `imagenes/estanque.png`. Coloca la imagen en la ruta "
            "`imagenes/estanque.png` a la misma altura que este archivo y vuelve a ejecutar."
        )
        st.code(
            "Estructura esperada:\n"
            "📁 tu_app/\n"
            " ├─ app.py (o main.py)\n"
            " ├─ pages/\n"
            " └─ imagenes/estanque.png"
        )

st.markdown("### Presiona un botón para elegir el modelo que desees analizar.")

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

st.write("     ")

# --------- Botones en columnas ----------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    if st.button("**Modelo 1**", help="Predice clorofila con datos del lago de Amatitlán", use_container_width=True):
        go("lago_amatitlan")

with c2:
    if st.button("**Modelo 2**", help="Predice clorofila y ficocianina con datos del lago de Atitlán", use_container_width=True):
        go("lago_atitlan")

with c3:
    if st.button("**Modelo 3**", help="Predice clorofila con datos del lago de Amatitlán y de Atitlán", use_container_width=True):
        go("ambos_lagos")
