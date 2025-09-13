# ============================================
# Página: Modelado y Clasificación (Clorofila)
# Lee data desde tus URLs de GitHub (raw),
# entrena/regresa/clasifica y ordena TODAS
# las figuras en pestañas, con banderas en sidebar.
# ============================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Evita ciertos warnings numéricos

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ==== TensorFlow/Keras (sección de regresión NN) ====
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib.patches import Rectangle

# ===========================
# Config de página y título
# ===========================
st.set_page_config(page_title="Dashboard cianobacteria — Modelos", layout="wide")
st.title("🧪 Dashboard cyanobacteria — Modelos y Clasificación")

# ===========================
# Rutas/URLs (lo que tenías)
# ===========================
EXCEL_ORIG_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/refs/heads/master/DATOS_AMSA.csv"
CSV_LIMPIO_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/refs/heads/master/datos_amsa.csv"
CSV_FILTRADO_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/refs/heads/master/datos_filtrados.csv"
PRED_REG_CSV_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/refs/heads/master/predicciones_clorofila.csv"

# Nombres de salida locales (para descargas desde la app)
MODEL_PATH = "modelo_clorofila.keras"
SCALER_PATH = "scaler_clorofila.pkl"
PRED_REG_CSV = "predicciones_clorofila_LOCAL.csv"
PRED_CLASES_DESDE_REG = "predicciones_clases_desde_regresion_LOCAL.csv"

# ===========================
# Columnas y opciones globales
# ===========================
columnas_entrada = [
    "pH",
    "Temperatura (°C)",
    "Conductividad (μS/cm)",
    "Oxígeno Disuelto (mg/L)",
    "Turbidez (NTU)"
]
columna_salida = "Clorofila (μg/L)"

# ======= Sidebar: Banderas =======
st.sidebar.header("⚙️ Controles")
RUN_TRAIN_NN = st.sidebar.checkbox("Entrenar red neuronal (regresión)", value=True)
RUN_CONFUSION_FROM_REGRESSION = st.sidebar.checkbox("Matriz de confusión desde regresión (umbrales)", value=True)
RUN_RF = st.sidebar.checkbox("Baseline: RandomForestRegressor", value=True)
RUN_KFOLD = st.sidebar.checkbox("KFold CV (NN regresión)", value=True)
RUN_CLF = st.sidebar.checkbox("Clasificación directa (SVM/KNN, 4 clases)", value=True)

st.sidebar.markdown("---")
USE_ROBUST_SCALER = st.sidebar.selectbox("Scaler NN", ["RobustScaler", "StandardScaler"]) == "RobustScaler"
Y_TRANSFORM = st.sidebar.selectbox("Transformación de y", ["log1p", "None"])
LOSS = st.sidebar.selectbox("Función de pérdida NN", ["huber", "mse"])

# ===========================
# Carga de datos (cache)
# ===========================
@st.cache_data(show_spinner=True)
def cargar_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

with st.expander("📥 Fuentes de datos (URLs)", expanded=False):
    st.write("**EXCEL_ORIG_URL** (CSV derivado del Excel Hoja3):", EXCEL_ORIG_URL)
    st.write("**CSV_LIMPIO_URL**:", CSV_LIMPIO_URL)
    st.write("**CSV_FILTRADO_URL**:", CSV_FILTRADO_URL)
    st.write("**PRED_REG_CSV_URL**:", PRED_REG_CSV_URL)

# Usaremos por defecto el CSV limpio como dataset principal
df = cargar_csv(CSV_LIMPIO_URL)

# ===========================
# Verificación/limpieza básicas
# ===========================
faltantes = [c for c in columnas_entrada + [columna_salida] if c not in df.columns]
if faltantes:
    st.error(f"Faltan columnas en el dataset: {faltantes}")
    st.stop()

# A numérico + outlier clipping defensivo
for col in columnas_entrada + [columna_salida]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in columnas_entrada:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

df = df.dropna(subset=columnas_entrada + [columna_salida]).reset_index(drop=True)

# ===========================
# Split base (compartido)
# ===========================
X = df[columnas_entrada].values
y_real = df[columna_salida].values

if Y_TRANSFORM == "log1p":
    y_trans = np.log1p(y_real)
else:
    y_trans = y_real.copy()

X_train, X_test, y_train_t, y_test_t = train_test_split(
    X, y_trans, test_size=0.20, random_state=42
)

# ===========================
# Utilidad: Matriz de confusión bonita → FIG
# ===========================
def plot_confusion_matrix_pretty(cm, labels, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    n = len(labels)
    ax.set_xlim(-0.5, n-0.5); ax.set_ylim(n-0.5, -0.5)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title(title)

    # Cuadrícula
    for i in range(n+1):
        ax.axhline(i-0.5, color="#888", lw=0.6, alpha=0.6)
        ax.axvline(i-0.5, color="#888", lw=0.6, alpha=0.6)

    # Pintar SOLO la diagonal
    for i in range(n):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, facecolor="#78c679", alpha=0.35, edgecolor="none"))

    # Números grandes
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", va="center", ha="center", fontsize=16, fontweight="bold")

    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.tight_layout()
    return fig

# ===========================
# Tabs para ordenar todas las figuras
# ===========================
tabs = st.tabs([
    "📈 Regresión NN",
    "🧩 Matriz desde Regresión",
    "🌲 Random Forest (baseline)",
    "🔁 K-Fold CV (NN)",
    "🎯 Clasificación directa (SVM/KNN)"
])

# ===========================
# 1) REGRESIÓN NN
# ===========================
with tabs[0]:
    st.subheader("📈 Regresión con Red Neuronal")

    if RUN_TRAIN_NN:
        scaler = RobustScaler() if USE_ROBUST_SCALER else StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        def build_model(input_dim: int) -> keras.Model:
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.15),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
            loss_fn = Huber(delta=1.0) if LOSS == "huber" else "mse"
            model.compile(optimizer=keras.optimizers.Adam(), loss=loss_fn, metrics=["mae"])
            return model

        model = build_model(X_train_s.shape[1])

        early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=1)

        with st.spinner("Entrenando la red neuronal..."):
            hist = model.fit(
                X_train_s, y_train_t,
                validation_split=0.20,
                epochs=600,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

        # Curva de pérdida
        fig_loss, ax = plt.subplots()
        ax.plot(hist.history["loss"], label="Pérdida entrenamiento")
        ax.plot(hist.history["val_loss"], label="Pérdida validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.set_title(f"Curva de entrenamiento (loss={LOSS}, y_transform={Y_TRANSFORM})")
        ax.grid(True); ax.legend(); fig_loss.tight_layout()
        st.pyplot(fig_loss, use_container_width=True)

        # Predicciones y des-transformación
        y_pred_train_t = model.predict(X_train_s, verbose=0).ravel()
        y_pred_test_t  = model.predict(X_test_s,  verbose=0).ravel()

        if Y_TRANSFORM == "log1p":
            y_true_test = np.expm1(y_test_t)
            y_pred_test = np.expm1(y_pred_test_t)
        else:
            y_true_test = y_test_t
            y_pred_test = y_pred_test_t

        mse  = mean_squared_error(y_true_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true_test, y_pred_test)
        r2   = r2_score(y_true_test, y_pred_test)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE (test)", f"{mse:.3f}")
        col2.metric("RMSE (test)", f"{rmse:.3f}")
        col3.metric("MAE (test)", f"{mae:.3f}")
        col4.metric("R² (test)", f"{r2:.3f}")

        # Guardado local + botón de descarga
        model.save(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            st.download_button("⬇️ Descargar modelo (.keras)", data=f, file_name=MODEL_PATH, mime="application/octet-stream")
        joblib.dump(scaler, SCALER_PATH)
        with open(SCALER_PATH, "rb") as f:
            st.download_button("⬇️ Descargar scaler (.pkl)", data=f, file_name=SCALER_PATH, mime="application/octet-stream")

        df_preds = pd.DataFrame({
            "Clorofila_real (μg/L)": y_true_test,
            "Clorofila_predicha (μg/L)": y_pred_test
        })
        csv_bytes = df_preds.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar predicciones (CSV)", data=csv_bytes, file_name=PRED_REG_CSV, mime="text/csv")

    else:
        st.info("Activa **Entrenar red neuronal (regresión)** en el panel lateral para ver esta sección.")

# ===========================
# 2) MATRIZ DESDE REGRESIÓN
# ===========================
with tabs[1]:
    st.subheader("🧩 Matriz de confusión (Regresión → Rangos)")

    if RUN_TRAIN_NN and RUN_CONFUSION_FROM_REGRESSION:
        bins = [0, 2, 7, 40, np.inf]
        labels_bins = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]

        y_true_clf_reg = pd.cut(y_true_test, bins=bins, labels=labels_bins, right=False)
        y_pred_clf_reg = pd.cut(y_pred_test,  bins=bins, labels=labels_bins, right=False)

        cm_reg = confusion_matrix(y_true_clf_reg, y_pred_clf_reg, labels=labels_bins)
        fig_cm = plot_confusion_matrix_pretty(cm_reg, labels_bins, "Matriz de confusión (Regresión → Rangos)")
        st.pyplot(fig_cm, use_container_width=True)

        # -------- FIX AQUÍ --------
        rep_reg = classification_report(
            y_true_clf_reg, y_pred_clf_reg,
            labels=labels_bins,        # fuerza las 4 clases en el orden dado
            target_names=labels_bins,  # nombres alineados
            digits=3,
            zero_division=0            # evita ValueError cuando una clase no aparece
        )
        st.code(rep_reg)
        # --------------------------

        # Descarga CSV de clases
        df_cls = pd.DataFrame({
            "Clorofila_real (µg/L)": y_true_test,
            "Clase_real": y_true_clf_reg.values,
            "Clorofila_predicha (µg/L)": y_pred_test,
            "Clase_predicha": y_pred_clf_reg.values
        })
        st.download_button("⬇️ Descargar clases desde regresión (CSV)",
                           data=df_cls.to_csv(index=False).encode("utf-8"),
                           file_name=PRED_CLASES_DESDE_REG,
                           mime="text/csv")
    else:
        st.info("Activa **Regresión NN** y **Matriz desde regresión** para visualizar.")

