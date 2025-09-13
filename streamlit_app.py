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
import pathlib, re, textwrap, sys

# --- Depurador: localiza cualquier ".values" peligroso en este archivo ---
try:
    src_path = pathlib.Path(__file__)
    src = src_path.read_text(encoding="utf-8")
    bads = [m.start() for m in re.finditer(r"y_(?:true|pred)_clf_reg\.values", src)]
    if bads:
        ln = src[:bads[0]].count("\n") + 1
        snippet = "\n".join(src.splitlines()[max(0, ln-4): ln+4])
        import streamlit as st
        st.error(f"⚠️ Hay una referencia a '.values' en la línea {ln} de {src_path.name}. "
                 "Reemplázala por pd.Series(...).astype('string').\n\n"
                 "Fragmento actual:\n" + snippet)
        st.stop()
except Exception:
    pass


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
st.caption("Los resultados obtenidos por el modelo se estarán visualizando en tiempo real en esta aplicación.")
st.info("🔖 Build check: v1.0.3")


# ===========================
# Rutas/URLs 
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
TRY_NEW_DATA = st.sidebar.toggle("Probar modelo con datos nuevos", value=True, key = "try_new")

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
    "🎯 Clasificación directa (SVM/KNN)",
    "🧐 Visualización de nuevas predicciones"
])
st.caption(f"Número de pestañas: {len(tabs)}")

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
    if RUN_TRAIN_NN and 'model' in locals():
        bins = [0, 2, 7, 40, np.inf]
        labels_bins = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]

        # Convertir a Series de tipo string (NO usar .values)
        y_true_bins = pd.Series(
            pd.cut(y_true_test, bins=bins, labels=labels_bins, right=False),
            dtype="string"
        )
        y_pred_bins = pd.Series(
            pd.cut(y_pred_test,  bins=bins, labels=labels_bins, right=False),
            dtype="string"
        )

        cm_reg = confusion_matrix(y_true_bins, y_pred_bins, labels=labels_bins)
        fig_cm = plot_confusion_matrix_pretty(cm_reg, labels_bins, "Matriz de confusión (Regresión → Rangos)")
        st.pyplot(fig_cm, use_container_width=True)

        rep_reg = classification_report(
            y_true_bins, y_pred_bins,
            labels=labels_bins, target_names=labels_bins,
            digits=3, zero_division=0
        )
        st.code(rep_reg)

        # CSV de clases — SIN .values
        df_cls_reg = pd.DataFrame({
            "Clorofila_real (µg/L)": y_true_test,
            "Clase_real": y_true_bins,
            "Clorofila_predicha (µg/L)": y_pred_test,
            "Clase_predicha": y_pred_bins,
        })
        st.download_button("⬇️ Descargar clases desde regresión (CSV)",
                           data=df_cls_reg.to_csv(index=False).encode("utf-8"),
                           file_name=PRED_CLASES_DESDE_REG,
                           mime="text/csv")
    else:
        st.info("Entrena la **Regresión NN** para habilitar esta pestaña.")

# ===========================
# 3) RANDOM FOREST (BASELINE)
# ===========================
with tabs[2]:
    st.subheader("🌲 Baseline: RandomForestRegressor")

    if RUN_RF:
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X, y_real, test_size=0.20, random_state=42
        )
        rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
        with st.spinner("Entrenando RandomForest..."):
            rf.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf.predict(X_test_rf)

        mse_rf  = mean_squared_error(y_test_rf, y_pred_rf)
        rmse_rf = np.sqrt(mse_rf)
        mae_rf  = mean_absolute_error(y_test_rf, y_pred_rf)
        r2_rf   = r2_score(y_test_rf, y_pred_rf)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE (test)", f"{mse_rf:.2f}")
        c2.metric("RMSE (test)", f"{rmse_rf:.2f}")
        c3.metric("MAE (test)", f"{mae_rf:.2f}")
        c4.metric("R² (test)", f"{r2_rf:.3f}")

        # Importancia de características
        imp = pd.Series(rf.feature_importances_, index=columnas_entrada).sort_values(ascending=False)
        fig_imp, ax = plt.subplots(figsize=(6,4))
        imp.plot(kind="bar", ax=ax)
        ax.set_title("Importancia de características (RF)")
        ax.set_ylabel("Importancia")
        ax.grid(True, axis="y", alpha=0.4)
        fig_imp.tight_layout()
        st.pyplot(fig_imp, use_container_width=True)

        st.dataframe(imp.reset_index().rename(columns={"index":"Feature", 0:"Importance"}), use_container_width=True)
    else:
        st.info("Activa **RandomForestRegressor** para visualizar.")

# ===========================
# 4) K-FOLD CV (NN REGRESIÓN)
# ===========================
with tabs[3]:
    st.subheader("🔁 Validación Cruzada (K=5) para NN de Regresión")

    if RUN_KFOLD:
        X_raw = X.copy()
        y_raw = y_real.copy()

        def nn_builder(input_dim):
            m = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.15),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
            m.compile(optimizer=keras.optimizers.Adam(), loss=Huber(), metrics=["mae"])
            return m

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = []

        progress = st.progress(0.0, text="Ejecutando K-Fold...")
        for fold, (tr_idx, te_idx) in enumerate(kf.split(X_raw), start=1):
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]

            if Y_TRANSFORM == "log1p":
                y_tr_t = np.log1p(y_tr)
                y_te_t = np.log1p(y_te)
            else:
                y_tr_t = y_tr.copy()
                y_te_t = y_te.copy()

            scaler_cv = RobustScaler() if USE_ROBUST_SCALER else StandardScaler()
            X_tr_s = scaler_cv.fit_transform(X_tr)
            X_te_s = scaler_cv.transform(X_te)

            model_cv = nn_builder(X_tr_s.shape[1])
            es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0)
            model_cv.fit(X_tr_s, y_tr_t, validation_split=0.2, epochs=300, batch_size=32, verbose=0, callbacks=[es, rl])

            y_pred_t = model_cv.predict(X_te_s, verbose=0).ravel()
            y_pred = np.expm1(y_pred_t) if Y_TRANSFORM == "log1p" else y_pred_t

            mse  = mean_squared_error(y_te, y_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_te, y_pred)
            r2   = r2_score(y_te, y_pred)
            metrics.append((fold, mse, rmse, mae, r2))

            progress.progress(fold/5.0, text=f"Fold {fold}/5 completado")

        df_cv = pd.DataFrame(metrics, columns=["Fold","MSE","RMSE","MAE","R2"])
        st.dataframe(df_cv.style.format({"MSE":"{:.2f}","RMSE":"{:.2f}","MAE":"{:.2f}","R2":"{:.3f}"}),
                     use_container_width=True)

        st.write("**Promedios ± std:**")
        st.write(f"- MSE : {df_cv['MSE'].mean():.2f} ± {df_cv['MSE'].std():.2f}")
        st.write(f"- RMSE: {df_cv['RMSE'].mean():.2f} ± {df_cv['RMSE'].std():.2f}")
        st.write(f"- MAE : {df_cv['MAE'].mean():.2f} ± {df_cv['MAE'].std():.2f}")
        st.write(f"- R²  : {df_cv['R2'].mean():.3f} ± {df_cv['R2'].std():.3f}")

        st.download_button("⬇️ Descargar métricas K-Fold (CSV)",
                           data=df_cv.to_csv(index=False).encode("utf-8"),
                           file_name="kfold_metrics.csv",
                           mime="text/csv")
    else:
        st.info("Activa **K-Fold CV (NN)** para visualizar.")

# ===========================
# 5) CLASIFICACIÓN DIRECTA (SVM/KNN, 4 clases)
# ===========================
with tabs[4]:
    st.subheader("🎯 Clasificación directa (SVM/KNN) — 4 clases")

    if RUN_CLF:
        bins = [0, 2, 7, 40, np.inf]
        labels_bins = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]

        # Etiquetas de clase desde y_real
        y_cls_all = pd.cut(y_real, bins=bins, labels=labels_bins, right=False)

        # Coherencia de splits con la NN
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X, y_real, test_size=0.20, random_state=42
        )
        y_train_cls = pd.cut(y_train_rf, bins=bins, labels=labels_bins, right=False)
        y_test_cls  = pd.cut(y_test_rf,  bins=bins, labels=labels_bins, right=False)

        # Pipelines
        svm_clf = make_pipeline(StandardScaler(),
                                SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42))
        knn_clf = make_pipeline(StandardScaler(),
                                KNeighborsClassifier(n_neighbors=7, weights="distance"))

        with st.spinner("Entrenando SVM y KNN..."):
            svm_clf.fit(X_train_rf, y_train_cls)
            knn_clf.fit(X_train_rf, y_train_cls)

        y_pred_svm = svm_clf.predict(X_test_rf)
        y_pred_knn = knn_clf.predict(X_test_rf)

        cm_svm = confusion_matrix(y_test_cls, y_pred_svm, labels=labels_bins)
        cm_knn = confusion_matrix(y_test_cls, y_pred_knn, labels=labels_bins)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_svm = plot_confusion_matrix_pretty(cm_svm, labels_bins, "Matriz de confusión — SVM (4 clases)")
            st.pyplot(fig_svm, use_container_width=True)

            # -------- FIX AQUÍ --------
            rep_svm = classification_report(
                y_test_cls, y_pred_svm,
                labels=labels_bins,
                target_names=labels_bins,
                digits=3,
                zero_division=0
            )
            st.code(rep_svm)
            # --------------------------

        with col_b:
            fig_knn = plot_confusion_matrix_pretty(cm_knn, labels_bins, "Matriz de confusión — KNN (4 clases)")
            st.pyplot(fig_knn, use_container_width=True)

            # -------- FIX AQUÍ --------
            rep_knn = classification_report(
                y_test_cls, y_pred_knn,
                labels=labels_bins,
                target_names=labels_bins,
                digits=3,
                zero_division=0
            )
            st.code(rep_knn)
            # --------------------------

    else:
        st.info("Activa **Clasificación directa (SVM/KNN)** para visualizar.")

# ===========================
# 6) 🧐 VISUALIZACIÓN DE NUEVAS PREDICCIONES
# ===========================
with tabs[5]:
    st.subheader("🧐 Visualización de nuevas predicciones")
    st.caption("Sube un CSV con **solo**: pH, Temperatura (°C), Conductividad (μS/cm), Oxígeno Disuelto (mg/L), Turbidez (NTU).")

    if not TRY_NEW_DATA:
        st.info("Activa **‘🧪 Probar modelo con datos nuevos’** en el panel lateral para habilitar esta pestaña.")
    else:
        # --- Modelo / scaler para inferencia ---
        model_infer = model if 'model' in locals() else None
        scaler_infer = scaler if 'scaler' in locals() else None

        if model_infer is None or scaler_infer is None:
            st.warning("No encuentro un modelo/scaler entrenados en esta sesión. Sube tus archivos guardados.")
            mdl_file = st.file_uploader("Modelo (.keras/.h5)", type=["keras", "h5", "hdf5"], key="mdl_new")
            scl_file = st.file_uploader("Scaler (.pkl)", type=["pkl"], key="scl_new")
            if mdl_file is not None and scl_file is not None:
                try:
                    model_infer = keras.models.load_model(mdl_file)
                    scaler_infer = joblib.load(scl_file)
                    st.success("Modelo y scaler cargados correctamente.")
                except Exception as e:
                    st.error(f"No se pudo cargar el modelo/scaler: {e}")

        # Solo seguimos si tenemos ambos
        if model_infer is not None and scaler_infer is not None:
            # --- Carga de CSV nuevo ---
            up_csv = st.file_uploader("Cargar CSV con **nuevos** datos (sin clorofila)", type=["csv"], key="csv_newdata")
            if up_csv is None:
                st.info("Sube un CSV para ver matrices y tabla de resultados.")
            else:
                # Lectura y validación
                df_new = pd.read_csv(up_csv)

                def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
                    mapping = {
                        "ph": "pH",
                        "temperatura": "Temperatura (°C)",
                        "temperatura (c)": "Temperatura (°C)",
                        "temp (°c)": "Temperatura (°C)",
                        "conductividad": "Conductividad (μS/cm)",
                        "conductividad (us/cm)": "Conductividad (μS/cm)",
                        "conductividad (µs/cm)": "Conductividad (μS/cm)",
                        "conductividad (μs/cm)": "Conductividad (μS/cm)",
                        "oxigeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
                        "oxígeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
                        "turbidez (ntu)": "Turbidez (NTU)",
                        "turbiedad (ntu)": "Turbidez (NTU)",
                    }
                    ren = {c: mapping.get(c.strip().lower(), c) for c in df.columns}
                    return df.rename(columns=ren)

                df_new = _normalize_cols(df_new)
                req = ["pH","Temperatura (°C)","Conductividad (μS/cm)","Oxígeno Disuelto (mg/L)","Turbidez (NTU)"]
                faltantes_new = [c for c in req if c not in df_new.columns]
                if faltantes_new:
                    st.error(f"Faltan columnas requeridas: {faltantes_new}")
                else:
                    for c in req:
                        df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
                    n0 = len(df_new)
                    df_new = df_new.dropna(subset=req).reset_index(drop=True)
                    if len(df_new) < n0:
                        st.warning(f"Se omitieron {n0 - len(df_new)} filas por valores no numéricos/NaN en las entradas.")

                    # --- Predicción con la NN de regresión ---
                    X_new = df_new[req].values
                    X_new_s = scaler_infer.transform(X_new)
                    y_pred_t = model_infer.predict(X_new_s, verbose=0).ravel()
                    y_pred = np.expm1(y_pred_t) if Y_TRANSFORM == "log1p" else y_pred_t
                    y_pred = np.clip(y_pred, 0.0, None)

                    # Rangos y etiquetas
                    BINS = [0, 2, 7, 40, np.inf]
                    LABELS = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]
                    cls_reg = pd.Series(pd.cut(y_pred, bins=BINS, labels=LABELS, right=False), dtype="string")

                    # Clasificadores directos (SVM/KNN)
                    if 'svm_clf' in locals() and 'knn_clf' in locals():
                        clf_svm, clf_knn = svm_clf, knn_clf
                    else:
                        y_all_cls = pd.cut(y_real, bins=BINS, labels=LABELS, right=False)
                        clf_svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale",
                                                                      class_weight="balanced", random_state=42))
                        clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
                        clf_svm.fit(X, y_all_cls)
                        clf_knn.fit(X, y_all_cls)

                    cls_svm = clf_svm.predict(X_new)
                    cls_knn = clf_knn.predict(X_new)

                    # Matrices
                    cm_reg_new = confusion_matrix(cls_reg, cls_reg, labels=LABELS)
                    cm_svm_new = confusion_matrix(cls_reg, cls_svm, labels=LABELS)
                    cm_knn_new = confusion_matrix(cls_reg, cls_knn, labels=LABELS)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.caption("Matriz (Regresión NN → Rangos)")
                        st.pyplot(plot_confusion_matrix_pretty(cm_reg_new, LABELS, "Regresión NN (rangos)"), use_container_width=True)
                    with c2:
                        st.caption("Matriz (SVM vs rangos de NN)")
                        st.pyplot(plot_confusion_matrix_pretty(cm_svm_new, LABELS, "SVM vs NN (proxy)"), use_container_width=True)
                    with c3:
                        st.caption("Matriz (KNN vs rangos de NN)")
                        st.pyplot(plot_confusion_matrix_pretty(cm_knn_new, LABELS, "KNN vs NN (proxy)"), use_container_width=True)

                    # Tabla + descarga
                    df_out = df_new.copy()
                    df_out["Clorofila_predicha (μg/L)"] = y_pred
                    df_out["Clase_NN"]  = cls_reg
                    df_out["Clase_SVM"] = cls_svm
                    df_out["Clase_KNN"] = cls_knn
                    st.success("¡Predicción sobre los nuevos datos lista!")
                    st.dataframe(df_out.head(50), use_container_width=True)
                    st.download_button(
                        "⬇️ Descargar CSV con nuevas predicciones",
                        data=df_out.to_csv(index=False).encode("utf-8"),
                        file_name="nuevas_predicciones_con_clases.csv",
                        mime="text/csv"
                    )

                    fig_hist, axh = plt.subplots()
                    axh.hist(y_pred, bins=30)
                    axh.set_title("Distribución de Clorofila predicha (μg/L)")
                    axh.set_xlabel("Clorofila (μg/L)")
                    axh.set_ylabel("Frecuencia")
                    st.pyplot(fig_hist, use_container_width=True)

    
