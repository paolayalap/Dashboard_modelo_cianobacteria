import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # opcional

import io, unicodedata, requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (mean_squared_error, mean_absolute_error, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ==============================
# Config general
# ==============================
st.set_page_config(page_title="🧪 Dashboard cyanobacteria", layout="wide")
st.title("🧪 Dashboard cyanobacteria")
st.info("Carga CSV (coma), limpieza, **Regresión (NN)** para clorofila y **Clasificación (SVM/KNN)** con umbral 40 µg/L.")

# Tus URLs (en crudo, sin refs/heads)
LAKE_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/datos_lago.csv"
NEW_URL  = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/dataframe.csv"
UMBRAL = 40.0
RANDOM_STATE = 42

# Archivos locales (si los subes al repo)
MODEL_PATH_KERAS = "modelo_clorofila.keras"  # opcional
MODEL_PATH_H5    = "modelo_clorofila.h5"     # opcional
SCALER_REG_PATH  = "scaler_clorofila.gz"     # opcional

# ==============================
# Utilidades
# ==============================
def _strip_accents_lower(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in s.lower() if ch.isalnum())

def best_match_column(df_cols, aliases):
    norm_map = {c: _strip_accents_lower(c) for c in df_cols}
    alias_norm = [_strip_accents_lower(a) for a in aliases]
    for raw, norm in norm_map.items():
        if norm in alias_norm:
            return raw
    return None

@st.cache_data(show_spinner=False)
def read_csv_commas_only(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, sep=",", encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(url, sep=",", encoding="latin-1", engine="python")

def try_load_regression_model():
    """Carga modelo de regresión si existe en el repo; devuelve (model, scaler) o (None, None)."""
    model = None
    scaler = None
    try:
        if os.path.exists(SCALER_REG_PATH):
            scaler = joblib.load(SCALER_REG_PATH)
        if os.path.exists(MODEL_PATH_KERAS):
            import tensorflow as tf  # import lazily
            model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        elif os.path.exists(MODEL_PATH_H5):
            import tensorflow as tf
            try:
                model = tf.keras.models.load_model(MODEL_PATH_H5)
            except TypeError:
                # cargar sin compilar y recompilar
                model = tf.keras.models.load_model(MODEL_PATH_H5, compile=False)
                model.compile(optimizer='adam',
                              loss=tf.keras.losses.MeanSquaredError(),
                              metrics=[tf.keras.metrics.MeanAbsoluteError()])
    except Exception as e:
        st.warning(f"No se pudo cargar modelo/scaler existentes: {e}")
    return model, scaler

# ==============================
# Carga de datos
# ==============================
with st.expander("Datos IBAGUA (CSV)", expanded=True):
    try:
        raw = read_csv_commas_only(LAKE_URL)
        st.success("CSV del lago cargado ✅")
        st.dataframe(raw.head(30), use_container_width=True)
        st.caption(f"Shape lago: {raw.shape[0]} × {raw.shape[1]}")
    except Exception as e:
        st.error("No se pudo leer el CSV del lago. Revisa separador por comas y encabezados.")
        st.exception(e)
        st.stop()

# ==============================
# Mapeo flexible de columnas
# ==============================
ALIASES = {
    "Clorofila (µg/L)": ["Clorofila (µg/L)", "Clorofila (ug/L)", "clorofila", "chlorophyll"],
    "Temperatura (°C)": ["Temperatura (°C)", "Temperatura (C)", "Temperatura", "temperature", "temp"],
    "pH": ["pH", "ph"],
    "Oxígeno Disuelto (mg/L)": ["Oxígeno Disuelto (mg/L)", "Oxigeno Disuelto (mg/L)", "DO (mg/L)", "OD (mg/L)", "dissolved oxygen"],
    "Turbidez (NTU)": ["Turbidez (NTU)", "Turbiedad (NTU)", "turbidez", "turbidity", "ntu"],
    "Conductividad (mS/cm)": ["Conductividad (mS/cm)", "Conductividad", "conductivity", "ms/cm", "mS/cm"],
}

auto_map = {}
for canonical, aliases in ALIASES.items():
    found = best_match_column(list(raw.columns), aliases)
    if found is not None:
        auto_map[canonical] = found

missing_for_auto = [k for k in ALIASES.keys() if k not in auto_map]
if missing_for_auto:
    st.warning("Algunas columnas no se detectaron automáticamente. Elige manualmente abajo.")

cols_list = list(raw.columns)
def picker(label, default_raw):
    idx = cols_list.index(default_raw) if default_raw in cols_list else 0
    return st.selectbox(label, cols_list, index=idx)

st.subheader("Mapeo de columnas")
col_map = {}
for canonical in ALIASES.keys():
    proposed = auto_map.get(canonical, None)
    col_map[canonical] = picker(f"{canonical}", proposed if proposed else cols_list[0])

if len(set(col_map.values())) < len(col_map.values()):
    st.error("La misma columna fue asignada a más de un campo. Corrige los selectores.")
    st.stop()

# ==============================
# Preprocesamiento común
# ==============================
df = raw.copy()
target_col = col_map["Clorofila (µg/L)"]
if target_col in df.columns:
    df[target_col] = df[target_col].replace("NR", 0)

df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

FEATURES = [
    col_map["Temperatura (°C)"],
    col_map["pH"],
    col_map["Oxígeno Disuelto (mg/L)"],
    col_map["Turbidez (NTU)"],
    col_map["Conductividad (mS/cm)"],
]

X = df[FEATURES]
y = df[target_col].copy().reset_index(drop=True)

# ==============================
# Tabs
# ==============================
tab_reg, tab_clf, tab_new = st.tabs(["🔵 Regresión (NN)", "🟠 Clasificación (SVM/KNN)", "🟢 Nuevos 50 (CSV)"])

# ---------- REGRESIÓN ----------
with tab_reg:
    st.subheader("Entrenamiento de red neuronal (regresión)")
    st.caption("Arquitectura: 5→64→32→16→1, pérdida MSE, métrica MAE. Se entrena con log1p(y).")

    # si hay scaler/modelo en repo, precargarlos
    pre_model, pre_scaler = try_load_regression_model()
    if pre_scaler is not None:
        scaler_reg = pre_scaler
    else:
        scaler_reg = StandardScaler()

    X_scaled = scaler_reg.fit_transform(X)

    col1, col2 = st.columns(2)
    with col1:
        st.write("X escalado (preview)")
        st.dataframe(pd.DataFrame(X_scaled, columns=FEATURES).head(10), use_container_width=True)
    with col2:
        st.write("Distribución de clorofila (µg/L)")
        fig_hist, ax = plt.subplots()
        ax.hist(y, bins=30)
        ax.set_xlabel("Clorofila (µg/L)"); ax.set_ylabel("Frecuencia")
        st.pyplot(fig_hist)

    X_ent, X_pru, y_ent, y_pru = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE)

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

    # Si hay modelo precargado y quieres usarlo sin entrenar:
    if pre_model is not None:
        st.success("Modelo de regresión precargado desde el repo ✅ (puedes re-entrenar si deseas)")
        st.session_state["reg_model"] = pre_model
        st.session_state["reg_scaler"] = scaler_reg

    if st.button("🚀 Entrenar/Actualizar modelo de regresión"):
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

            st.session_state["reg_model"] = model
            st.session_state["reg_scaler"] = scaler_reg

            y_pred_log = model.predict(X_pru)
            y_pred = np.expm1(y_pred_log).ravel()
            mse = mean_squared_error(y_pru, y_pred)
            mae = mean_absolute_error(y_pru, y_pred)

            st.success(f"Listo ✅  |  MSE: {mse:.2f}  |  MAE: {mae:.2f}")

            fig_loss, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Train loss')
            ax.plot(history.history['val_loss'], label='Val loss')
            ax.set_xlabel('Épocas'); ax.set_ylabel('MSE (log1p)')
            ax.set_title('Evolución del error'); ax.legend()
            st.pyplot(fig_loss)

            fig_scatter, ax2 = plt.subplots()
            ax2.scatter(y_pru, y_pred, s=14)
            lo = float(np.min([y_pru.min(), y_pred.min()]))
            hi = float(np.max([y_pru.max(), y_pred.max()]))
            ax2.plot([lo, hi], [lo, hi])
            ax2.set_xlabel('Real (µg/L)'); ax2.set_ylabel('Predicho (µg/L)')
            ax2.set_title('Clorofila: Real vs Predicho')
            st.pyplot(fig_scatter)
    else:
        st.info("Puedes usar el modelo precargado (si existe) o entrenar uno nuevo.")

# ---------- CLASIFICACIÓN ----------
with tab_clf:
    st.subheader("Clasificación por umbral (40 µg/L)")
    y_cls = (y > UMBRAL).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, class_weight='balanced',
                                                  C=1.0, gamma='scale', random_state=RANDOM_STATE))
    knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights='distance'))

    if st.button("📊 Validación cruzada SVM/KNN"):
        with st.spinner("Computando CV..."):
            y_pred_svm_cv  = cross_val_predict(svm_pipe, X.values, y_cls.values, cv=cv)
            y_proba_svm_cv = cross_val_predict(svm_pipe, X.values, y_cls.values, cv=cv, method='predict_proba')[:, 1]
            cm_svm = confusion_matrix(y_cls, y_pred_svm_cv, labels=[0,1])

            fig1, ax1 = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Bajo (<40)', 'Alto (>40)']).plot(ax=ax1, values_format='d')
            ax1.set_title('SVM (CV) - Matriz de confusión')
            st.pyplot(fig1)
            st.caption(f"SVM (CV) ROC-AUC: {roc_auc_score(y_cls, y_proba_svm_cv):.3f}")
            st.text(classification_report(y_cls, y_pred_svm_cv, target_names=['Bajo (<40)','Alto (>40)'], zero_division=0))

            y_pred_knn_cv  = cross_val_predict(knn_pipe, X.values, y_cls.values, cv=cv)
            cm_knn = confusion_matrix(y_cls, y_pred_knn_cv, labels=[0,1])

            fig2, ax2 = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['Bajo (<40)', 'Alto (>40)']).plot(ax=ax2, values_format='d')
            ax2.set_title('KNN (CV) - Matriz de confusión')
            st.pyplot(fig2)

    X_tr, X_te, y_tr, y_te = train_test_split(X.values, y_cls.values, test_size=0.4, random_state=RANDOM_STATE, stratify=y_cls.values)
    scaler_clf = StandardScaler().fit(X_tr)
    X_tr_s = scaler_clf.transform(X_tr)
    X_te_s = scaler_clf.transform(X_te)

    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, gamma='scale', random_state=RANDOM_STATE).fit(X_tr_s, y_tr)
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance').fit(X_tr_s, y_tr)

    def eval_and_plot(model, Xs, y_true, title_prefix):
        y_pred = model.predict(Xs)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bajo (<40)','Alto (>40)']).plot(ax=ax, values_format='d')
        ax.set_title(f'{title_prefix} - Matriz de confusión (hold-out)')
        st.pyplot(fig)

        prc, rcl, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
        try:
            auc = roc_auc_score(y_true, model.predict_proba(Xs)[:,1])
        except Exception:
            auc = np.nan
        st.write(f"**{title_prefix}** → Precision[Bajo,Alto]={prc[0]:.3f}, {prc[1]:.3f} | "
                 f"Recall[Bajo,Alto]={rcl[0]:.3f}, {rcl[1]:.3f} | F1[Bajo,Alto]={f1[0]:.3f}, {f1[1]:.3f} | ROC-AUC={auc:.3f}")

    st.markdown("**Evaluación hold-out (40%)**")
    eval_and_plot(svm, X_te_s, y_te, "SVM")
    eval_and_plot(knn, X_te_s, y_te, "KNN")

    st.session_state["clf_scaler"] = scaler_clf
    st.session_state["clf_svm"] = svm
    st.session_state["clf_knn"] = knn

# ---------- NUEVOS 50 ----------
with tab_new:
    st.subheader("Nuevos 50 registros (CSV)")
    st.caption("Se usa NEW_URL por defecto; también puedes subir un CSV propio.")

    use_repo_new = st.checkbox("Usar CSV del repo (NEW_URL)", value=True)
    nuevo = None
    if use_repo_new:
        try:
            nuevo = read_csv_commas_only(NEW_URL)
            st.success("CSV nuevo cargado desde el repo ✅")
        except Exception as e:
            st.error("No se pudo leer el NEW_URL.")
            st.exception(e)
    else:
        file = st.file_uploader("Sube CSV separado por comas", type=["csv"])
        if file is not None:
            try:
                nuevo = pd.read_csv(file, sep=",", encoding="utf-8")
            except Exception:
                file.seek(0)
                nuevo = pd.read_csv(file, sep=",", encoding="latin-1")

    if nuevo is not None:
        # mapear columnas del nuevo dataset usando el mismo col_map
        for c in FEATURES:
            nuevo[c] = pd.to_numeric(nuevo.get(c, np.nan), errors="coerce")

        df50 = nuevo[FEATURES].head(50).dropna().copy()
        st.write(f"Filas válidas entre los primeros 50: **{len(df50)}**")
        st.dataframe(df50.head(10), use_container_width=True)

        # Predicción de clorofila (modelo de la sesión o del repo)
        reg_model = st.session_state.get("reg_model", None)
        reg_scaler = st.session_state.get("reg_scaler", None)

        if reg_model is None or reg_scaler is None:
            # intentar cargar desde archivos del repo
            m, s = try_load_regression_model()
            if m is not None and s is not None:
                reg_model, reg_scaler = m, s

        out = df50.copy()

        if reg_model is not None and reg_scaler is not None:
            X50_reg_scaled = reg_scaler.transform(df50.values)
            try:
                y50_log = reg_model.predict(X50_reg_scaled)
            except Exception:
                # si el modelo fue entrenado sin log1p, igualmente devolvemos
                y50_log = reg_model.predict(X50_reg_scaled)
            chl50 = np.expm1(y50_log).ravel()
            chl50 = np.clip(chl50, 0, None)
            out["Clorofila predicha (µg/L)"] = chl50

            # gráficas rápidas
            figA, axA = plt.subplots()
            axA.plot(range(len(chl50)), chl50)
            axA.set_xlabel('Índice (0..N)'); axA.set_ylabel('Clorofila predicha (µg/L)')
            axA.set_title('Serie: clorofila predicha (primeros 50)')
            st.pyplot(figA)

            figB, axB = plt.subplots()
            axB.hist(chl50, bins=15)
            axB.set_xlabel('Clorofila predicha (µg/L)'); axB.set_ylabel('Frecuencia')
            axB.set_title('Distribución de clorofila predicha')
            st.pyplot(figB)

            cls_from_reg = (chl50 > UMBRAL).astype(int)
            out["Clase por regresión (>40)"] = np.where(cls_from_reg==1, "Alto (>40)", "Bajo (<40)")
        else:
            st.info("Entrena o provee un modelo de regresión para obtener clorofila predicha.")

        # Clasificación con SVM/KNN si entrenados en esta sesión
        if "clf_scaler" in st.session_state and "clf_svm" in st.session_state and "clf_knn" in st.session_state:
            X50_clf_scaled = st.session_state["clf_scaler"].transform(df50.values)
            cls50_svm = st.session_state["clf_svm"].predict(X50_clf_scaled)
            cls50_knn = st.session_state["clf_knn"].predict(X50_clf_scaled)
            out["Clase SVM"] = np.where(cls50_svm==1, "Alto (>40)", "Bajo (<40)")
            out["Clase KNN"] = np.where(cls50_knn==1, "Alto (>40)", "Bajo (<40)")
        else:
            st.info("Entrena primero los clasificadores en la pestaña naranja para etiquetar SVM/KNN.")

        # Mostrar y permitir descarga
        st.subheader("Resultados (primeros 50)")
        st.dataframe(out, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 Descargar resultados (CSV)",
            data=csv_bytes,
            file_name="predicciones_dataframe_clf.csv",
            mime="text/csv"
        )

# ==============================
# Requisitos sugeridos
# ==============================
st.caption("requirements.txt: streamlit, pandas, numpy, scikit-learn, matplotlib, requests, joblib, tensorflow-cpu")

