import streamlit as st
import pandas as pd
import io, requests

# ======================
#  Config
# ======================
URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/datos_lago.csv"

@st.cache_data(show_spinner=False)
def read_csv_robusto(url: str) -> pd.DataFrame:
    """
    Lee un CSV desde una URL probando encodings y separadores comunes.
    Como último recurso, descarga bytes y reemplaza caracteres inválidos.
    """
    # 1) Intentos rápidos
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (",", ";", "\t", "|"):
            try:
                return pd.read_csv(url, encoding=enc, sep=sep, engine="python")
            except Exception:
                continue

    # 2) Último recurso: descargar manualmente y decodificar
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = io.BytesIO(resp.content)

    try:
        # pandas >= 1.5 soporta encoding_errors
        return pd.read_csv(data, engine="python", sep=None, encoding="utf-8", encoding_errors="replace")
    except TypeError:
        # pandas más antiguo
        data.seek(0)
        text = resp.content.decode("latin-1", errors="replace")
        return pd.read_csv(io.StringIO(text), engine="python", sep=None)

# ======================
#  UI
# ======================
st.title('🧪 Dashboard cyanobacteria')

st.info('This is an app to demonstrate and analyze the data to create a system to detect cyanobacteria in water bodies with machine learning.')
st.write('First, we are going to present our reference data. This is going to have parameters like: pH, temperature, dissolved oxygen, conductivity and turbidity.')

with st.expander('Data IBAGUA'):
    st.write('**In 2017 and 2019, Evelyn Rodas took some measurements of chlorophyll-a to estimate phytoplankton in Lake Amatitlán. She shared this information so we could build our first models.**')

    try:
        df = read_csv_robusto(URL)
        st.success('CSV loaded successfully ✅')
        st.dataframe(df.head(50))   # muestra primeras filas para no saturar
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error("There was a problem reading the CSV file.")
        st.exception(e)
