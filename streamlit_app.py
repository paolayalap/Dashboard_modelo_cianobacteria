import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # opcional

import io
import unicodedata
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================================
# Config general
# =========================================
st.set_page_config(page_title="🧪 Dashboard cyanobacteria", layout="wide")
st.title("🧪 Dashboard cyanobacteria ")
st.info("This app loads CSV data, cleans it, and trains a NN model to estimate chlorophyll-a.")

# Tu CSV (separado por comas) en GitHub:
CSV_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/datos_lago.csv"

# =========================================
# Utilidades
# =========================================
def _strip_accents_lower(s: str) -> str:
    """minúsculas + sin acentos + solo alfanumérico (col names robustos)"""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # deja solo alfanumérico (une palabras)
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)

def best_match_column(df_cols, aliases):
    """Encuentra la primera columna en dfCols que haga match con la lista de aliases (normalizados)."""
    norm_map = {c: _strip_accents_lower(c) for c in df_cols}
    alias_norm = [_strip_accents_lower(a) for a in aliases]
    for raw, norm in norm_map.items():
        if norm in alias_norm:
            return raw
    return None

# =========================================
# Carga CSV (solo coma, encoding robusto)
# =========================================
@st.cache_data(show_spinner=False)
def read_csv_commas_only(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, sep=",", encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(url, sep=",", encoding="latin-1", engine="python")

with st.expander("Data IBAGUA (CSV)", expanded=True):
    try:
        raw = read_csv_commas_only(CSV_URL)
        st.success("CSV cargado ✅")
        st.dataframe(raw.head(30), use_container_width=True)
        st.caption(f"Shape: {raw.shape[0]} filas × {raw.shape[1]} columnas")
    except Exception as e:
        st.error("No se pudo leer el CSV. Verifica que esté separado por comas y con encabezados.")
        st.exception(e)
        st.stop()

# =========================================
# Mapeo flexible de columnas (aliases + UI)
# =========================================
# Objetivo y features con alias comunes
ALIASES = {
    # Target
    "Clorofila (µg/L)": [
        "Clorofila (µg/L)", "Clorofila (ug/L)", "clorofila", "chlorophyll", "chlorofila ug l"
    ],
    # Features
    "Temperatura (°C)": [
        "Temperatura (°C)", "Temperatura (C)", "Temperatura", "temperature", "temp", "temp (c)"
    ],
    "pH": [
        "pH", "ph"
    ],
    "Oxígeno Disuelto (mg/L)": [
        "Oxígeno Disuelto (mg/L)", "Oxigeno Disuelto (mg/L)", "DO (mg/L)", "OD (mg/L)",
        "oxigeno disuelto mg l", "do", "oxygen dissolved", "dissolved oxygen"
    ],
    "Turbidez (NTU)": [
        "Turbidez (NTU)", "Turbiedad (NTU)", "turbidez", "turbidity", "ntu"
    ],
    "Conductividad (mS/cm)": [
        "Conductividad (mS/cm)", "Conductividad", "conductivity", "ms/cm", "mS/cm"
    ],
}

# Intento de mapeo automático
auto_map = {}
for canonical, aliases in ALIASES.items():
    found = best_match_column(list(raw.columns), aliases)
    if found is not None:
        auto_map[canonical] = found

missing_for_auto = [k for k in ALIASES.keys() if k not in auto_map]

st.subheader("Mapeo de columnas")
if missing_for_auto:
    st.warning(
        "No pude reconocer automáticamente algunas columnas. "
        "Selecciona manualmente en los menús desplegables."
    )

# UI para confirmar/corregir el mapeo
cols_list = list(raw.columns)
def picker(label, default_raw):
    idx = cols_list.index(default_raw) if default_raw in cols_list else 0
    return st.selectbox(label, cols_list, index=idx)

col_map = {}
for canonical in ALIASES.keys():
    proposed = auto_map.get(canonical, None)
    col_map[canonical] = picker(f"{canonical}", proposed if proposed else cols_list[0])

# Validación final
mapped_names = list(col_map.values())
if len(set(mapped_names)) < len(mapped_names):
    st.error("Has asignado la misma columna del CSV a más de un campo. Corrige los selectores.")
    st.stop()

# =========================================
# Preprocesamiento
# =========================================
df = raw.copy()

TARGET_COL = col_map["Clorofila (µg/L)"]
# NR->0 solo en columna objetivo (si existiera)
if TARGET_COL in df.columns:
    df[TARGET_COL] = df[TARGET_COL].replace("NR", 0)

# Fuerza numérico y elimina filas con NaN
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna().reset_index(drop=True)

FEATURES = [
    col_map["Temperatura (°C)"],
    col_map["pH"],
    col_map["Oxígeno Disuelto (mg/L)"],
    col_map["Turbidez (NTU)"],
    col_map["Conductividad (mS/cm)"],
]

X = df[FEATURES]
y = df[TARGET_COL].copy().reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

col1, col2 = st.columns(2)
with col1:
    st.subheader("X escalado (preview)")
    st.dataframe(pd.DataFrame(X_scaled, columns=FEATURES).head(10), use_container_width=True)
with col2:
    st.subheader("Distribución de clorofila (µg/L)")
    fig_hist, ax = plt.subplots()
    ax.hist(y, bins=30)
    ax.set_xlabel("Clorofila (µg/L)")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig_hist)

# =========================================
# Split y entrenamiento (TF perezoso)
# =========================================
X_ent, X_pru, y_ent, y_pru = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.subheader("Entrenamiento del modelo")
st.caption("Arquitectura: 5→64→32→16→1, pérdida MSE, métrica MAE. Se entrena con y en log1p para estabilizar escala.")

def build_model(keras):
    model = keras.Sequential([
        keras.layers.Input(shape=(len(FEATURES),)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

if st.button("🚀 Entrenar modelo con CSV"):
    with st.spinner("Importando TensorFlow y entrenando..."):
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras
        except Exception as e:
            st.error("No se pudo importar TensorFlow. Revisa `requirements.txt` y `runtime.txt`.")
            st.exception(e)
            st.stop()

        y_ent_log = np.log1p(y_ent)
        model = build_model(keras)
        history = model.fit(
            X_ent, y_ent_log,
            validation_split=0.2,
            epochs=300,
            batch_size=8,
            verbose=0
        )

        # Predicciones y métricas (deslog)
        y_pred_log = model.predict(X_pru)
        y_pred = np.expm1(y_pred_log).ravel()
        mse = mean_squared_error(y_pru, y_pred)
        mae = mean_absolute_error(y_pru, y_pred)

        st.success(f"Listo ✅  |  MSE: {mse:.2f}  |  MAE: {mae:.2f}")

        # Curvas de pérdida
        fig_loss, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train loss')
        ax.plot(history.history['val_loss'], label='Val loss')
        ax.set_xlabel('Épocas'); ax.set_ylabel('MSE (log1p)')
        ax.set_title('Evolución del error'); ax.legend()
        st.pyplot(fig_loss)

        # Real vs. Predicho
        fig_scatter, ax2 = plt.subplots()
        ax2.scatter(y_pru, y_pred, s=14)
        lo = float(np.min([y_pru.min(), y_pred.min()]))
        hi = float(np.max([y_pru.max(), y_pred.max()]))
        ax2.plot([lo, hi], [lo, hi])
        ax2.set_xlabel('Real (µg/L)'); ax2.set_ylabel('Predicho (µg/L)')
        ax2.set_title('Clorofila: Real vs Predicho')
        st.pyplot(fig_scatter)
else:
    st.warning("Pulsa **Entrenar modelo con CSV** para generar métricas y gráficas.")

st.caption("requirements.txt: streamlit, pandas, numpy, scikit-learn, tensorflow-cpu, matplotlib, requests")
