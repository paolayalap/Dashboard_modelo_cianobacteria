import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # opcional

import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

# ======================
# Config
# ======================
st.set_page_config(page_title="🧪 Dashboard cyanobacteria", layout="wide")
st.title("🧪 Dashboard cyanobacteria")
st.info("This app loads CSV data (comma-separated), cleans it, and trains a NN model to estimate chlorophyll-a.")

CSV_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/datos_lago.csv"  # tu CSV en GitHub (coma)

# ======================
# Carga CSV (solo coma)
# ======================
@st.cache_data(show_spinner=False)
def read_csv_commas_only(url: str) -> pd.DataFrame:
    # Intenta UTF-8 y luego Latin-1, siempre con sep=","
    try:
        return pd.read_csv(url, sep=",", encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(url, sep=",", encoding="latin-1", engine="python")

with st.expander("Data IBAGUA (CSV)", expanded=True):
    try:
        raw = read_csv_commas_only(CSV_URL)
        st.success("CSV cargado ✅ (coma-separado)")
        st.dataframe(raw.head(30), use_container_width=True)
        st.caption(f"Shape: {raw.shape[0]} filas × {raw.shape[1]} columnas")
    except Exception as e:
        st.error("No se pudo leer el CSV. Verifica que esté separado por comas y con encabezados.")
        st.exception(e)
        st.stop()

# ======================
# Preprocesamiento
# ======================
df = raw.copy()

# Reemplaza NR -> 0 solo en clorofila (ajusta al nombre exacto de tu columna)
TARGET_COL = "Clorofila (µg/L)"
if TARGET_COL in df.columns:
    df[TARGET_COL] = df[TARGET_COL].replace("NR", 0)

# Fuerza numérico en todo y elimina filas con NaN
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna().reset_index(drop=True)

# Entradas/salida (ajusta nombres si difieren en tu CSV)
FEATURES = ['Temperatura (°C)', 'pH', 'Oxígeno Disuelto (mg/L)', 'Turbidez (NTU)', 'Conductividad (mS/cm)']
missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el CSV: {missing}")
    st.stop()

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

# ======================
# Train/test split y modelo
# ======================
X_ent, X_pru, y_ent, y_pru = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.subheader("Entrenamiento del modelo")
st.caption("Arquitectura: 5→64→32→16→1, pérdida MSE, métrica MAE. Se entrena con y en log1p para estabilizar escala.")

def build_model():
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
    with st.spinner("Entrenando..."):
        y_ent_log = np.log1p(y_ent)
        model = build_model()
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
        ax.set_xlabel('Épocas')
        ax.set_ylabel('MSE (log1p)')
        ax.set_title('Evolución del error')
        ax.legend()
        st.pyplot(fig_loss)

        # Real vs. Predicho
        fig_scatter, ax2 = plt.subplots()
        ax2.scatter(y_pru, y_pred, s=14)
        lo, hi = float(np.min([y_pru.min(), y_pred.min()])), float(np.max([y_pru.max(), y_pred.max()]))
        ax2.plot([lo, hi], [lo, hi])
        ax2.set_xlabel('Real (µg/L)')
        ax2.set_ylabel('Predicho (µg/L)')
        ax2.set_title('Clorofila: Real vs Predicho')
        st.pyplot(fig_scatter)
else:
    st.warning("Pulsa **Entrenar modelo con CSV** para generar métricas y gráficas.")

st.caption("requirements.txt: streamlit, pandas, numpy, scikit-learn, tensorflow, matplotlib, requests")
