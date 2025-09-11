import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # opcional

import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests, joblib

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
st.write("Carga CSV (coma), limpieza, **Regresión (NN)** para clorofila y **Clasificación (SVM/KNN)** con umbral 40 µg/L.")

# URLs crudos de GitHub
LAKE_URL = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/datos_lago.csv"
NEW_URL  = "https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/master/dataframe.csv"
UMBRAL = 40.0
RANDOM_STATE = 42

# Archivos guardados en el repo (se crean al entrenar)
MODEL_PATH_H5    = "modelo_clorofila.h5"
SCALER_REG_PATH  = "scaler_clorofila.gz"

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

def fig_small():
    # tamaño pensado para 1/3 de pantalla dentro de st.columns(3)
    return plt.subplots(figsize=(5.0, 3.4))

# ==============================
# 1) Datos y limpieza
# ==============================
with st.expander("📦 Datos IBAGUA (CSV) y limpieza", expanded=True):
    try:
        raw = read_csv_commas_only(LAKE_URL)
        st.success("CSV del lago cargado ✅")
    except Exception as e:
        st.error("No se pudo leer el CSV del lago.")
        st.exception(e)
        st.stop()

    # Mapeo flexible de columnas
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

    if len(auto_map) < len(ALIASES):
        st.warning("Algunas columnas no se detectaron automáticamente. Elige manualmente:")
    cols_list = list(raw.columns)
    def picker(label, default_raw):
        idx = cols_list.index(default_raw) if default_raw in cols_list else 0
        return st.selectbox(label, cols_list, index=idx)

    col_map = {}
    for canonical in ALIASES.keys():
        proposed = auto_map.get(canonical, None)
        col_map[canonical] = picker(f"{canonical}", proposed if proposed else cols_list[0])

    if len(set(col_map.values())) < len(col_map.values()):
        st.error("La misma columna fue asignada a más de un campo. Corrige los selectores.")
        st.stop()

    # Limpieza
    df = raw.copy()
    target_col = col_map["Clorofila (µg/L)"]
    df[target_col] = df[target_col].replace("NR", 0)  # solo objetivo

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

    # Resumen ordenado
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas limpias", len(df))
    c2.metric("Variables de entrada", len(FEATURES))
    c3.metric("Columna objetivo", target_col)

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    # Botón para generar todas las gráficas de datos
    if st.button("📊 Generar gráficas de datos (todas)"):
        # Hist y box del target + correlación simple
        a, b, c = st.columns(3)

        with a:
            fig, ax = fig_small()
            ax.hist(y, bins=30)
            ax.set_title("Hist: Clorofila (µg/L)")
            ax.set_xlabel("µg/L"); ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

        with b:
            fig, ax = fig_small()
            ax.boxplot([df[f] for f in FEATURES], labels=[f.replace("(µg/L)","") for f in FEATURES], vert=True, showfliers=False)
            ax.set_title("Boxplots (features)")
            st.pyplot(fig)

        with c:
            corr = df[FEATURES + [target_col]].corr(numeric_only=True)
            fig, ax = fig_small()
            im = ax.imshow(corr, aspect="auto")
            ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
            ax.set_yticks(range(len(corr.index)));  ax.set_yticklabels(corr.index)
            ax.set_title("Matriz de correlación")
            fig.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig)

# ==============================
# 2) Tabs: Regresión y Clasificación
# ==============================
tab_reg, tab_clf = st.tabs(["🔵 Regresión (NN)", "🟠 Clasificación (SVM/KNN)"])

# ---------- REGRESIÓN ----------
with tab_reg:
    st.subheader("Entrenamiento (NN) y gráficas")

    scaler_reg = StandardScaler()
    X_scaled = scaler_reg.fit_transform(X)
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

    if st.button("🚀 Entrenar modelo de regresión (y guardar)"):
        with st.spinner("Importando TensorFlow y entrenando..."):
            try:
                import tensorflow as tf  # noqa: F401
                from tensorflow import keras
            except Exception as e:
                st.error("No se pudo importar TensorFlow. Revisa requirements/runtime.")
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

            # Guardar a disco para próximos deploys
            model.save(MODEL_PATH_H5)
            joblib.dump(scaler_reg, SCALER_REG_PATH)

            # Guardar en sesión también
            st.session_state["reg_model"] = model
            st.session_state["reg_scaler"] = scaler_reg

            # Predicciones y métricas
            y_pred_log = model.predict(X_pru)
            y_pred = np.expm1(y_pred_log).ravel()
            mse = mean_squared_error(y_pru, y_pred)
            mae = mean_absolute_error(y_pru, y_pred)
            st.success(f"Entrenado y guardado ✅ | MSE: {mse:.2f} | MAE: {mae:.2f}")

            # === TODAS LAS GRÁFICAS (3 por fila) ===
            r1, r2, r3 = st.columns(3)

            with r1:
                fig, ax = fig_small()
                ax.plot(history.history['loss'], label='Train')
                ax.plot(history.history['val_loss'], label='Val')
                ax.set_title('Evolución del error')
                ax.set_xlabel('Épocas'); ax.set_ylabel('MSE (log1p)')
                ax.legend()
                st.pyplot(fig)

            with r2:
                fig, ax = fig_small()
                ax.scatter(y_pru, y_pred, s=14)
                lo = float(np.min([y_pru.min(), y_pred.min()]))
                hi = float(np.max([y_pru.max(), y_pred.max()]))
                ax.plot([lo, hi], [lo, hi])
                ax.set_title('Real vs Predicho')
                ax.set_xlabel('Real (µg/L)'); ax.set_ylabel('Predicho (µg/L)')
                st.pyplot(fig)

            with r3:
                fig, ax = fig_small()
                ax.hist(y_pred, bins=30)
                ax.set_title('Hist: y_pred (µg/L)')
                ax.set_xlabel('µg/L'); ax.set_ylabel('Frecuencia')
                st.pyplot(fig)

# ---------- CLASIFICACIÓN ----------
with tab_clf:
    st.subheader("Clasificación por umbral (40 µg/L) y gráficas")

    y_cls = (y > UMBRAL).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X.values, y_cls.values, test_size=0.4,
                                              random_state=RANDOM_STATE, stratify=y_cls.values)
    scaler_clf = StandardScaler().fit(X_tr)
    X_tr_s = scaler_clf.transform(X_tr)
    X_te_s = scaler_clf.transform(X_te)

    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, gamma='scale', random_state=RANDOM_STATE).fit(X_tr_s, y_tr)
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance').fit(X_tr_s, y_tr)

    st.session_state["clf_scaler"] = scaler_clf
    st.session_state["clf_svm"] = svm
    st.session_state["clf_knn"] = knn

    if st.button("📊 Generar gráficas de clasificación (todas)"):
        # 1) Validación cruzada (SVM + KNN)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, class_weight='balanced',
                                                       C=1.0, gamma='scale', random_state=RANDOM_STATE))
        knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights='distance'))

        y_pred_svm_cv  = cross_val_predict(svm_pipe, X.values, y_cls.values, cv=cv)
        y_proba_svm_cv = cross_val_predict(svm_pipe, X.values, y_cls.values, cv=cv, method='predict_proba')[:, 1]
        y_pred_knn_cv  = cross_val_predict(knn_pipe, X.values, y_cls.values, cv=cv)

        cm_svm = confusion_matrix(y_cls, y_pred_svm_cv, labels=[0,1])
        cm_knn = confusion_matrix(y_cls, y_pred_knn_cv, labels=[0,1])

        g1, g2, g3 = st.columns(3)

        with g1:
            fig, ax = fig_small()
            ConfusionMatrixDisplay(cm_svm, display_labels=['Bajo (<40)', 'Alto (>40)']).plot(ax=ax, values_format='d')
            ax.set_title('SVM (CV) - Confusión')
            st.pyplot(fig)

        with g2:
            fig, ax = fig_small()
            ConfusionMatrixDisplay(cm_knn, display_labels=['Bajo (<40)', 'Alto (>40)']).plot(ax=ax, values_format='d')
            ax.set_title('KNN (CV) - Confusión')
            st.pyplot(fig)

        with g3:
            # Reporte SVM texto
            rep = classification_report(y_cls, y_pred_svm_cv, target_names=['Bajo (<40)','Alto (>40)'], zero_division=0)
            st.markdown("**SVM (CV) – ROC-AUC:** {:.3f}".format(roc_auc_score(y_cls, y_proba_svm_cv)))
            st.text(rep)

        # 2) Hold-out (SVM + KNN)
        h1, h2, h3 = st.columns(3)

        def plot_holdout(model, Xs, y_true, titulo):
            fig, ax = fig_small()
            cm = confusion_matrix(y_true, model.predict(Xs), labels=[0,1])
            ConfusionMatrixDisplay(cm, display_labels=['Bajo (<40)','Alto (>40)']).plot(ax=ax, values_format='d')
            ax.set_title(titulo)
            return fig

        with h1:
            st.pyplot(plot_holdout(svm, X_te_s, y_te, "SVM (hold-out)"))

        with h2:
            st.pyplot(plot_holdout(knn, X_te_s, y_te, "KNN (hold-out)"))

        with h3:
            prc, rcl, f1, _ = precision_recall_fscore_support(y_te, svm.predict(X_te_s), labels=[0,1], zero_division=0)
            try:
                auc = roc_auc_score(y_te, svm.predict_proba(X_te_s)[:,1])
            except Exception:
                auc = np.nan
            st.markdown(f"**SVM hold-out** → Precision[Bajo,Alto]={prc[0]:.3f}, {prc[1]:.3f} | "
                        f"Recall[Bajo,Alto]={rcl[0]:.3f}, {rcl[1]:.3f} | F1[Bajo,Alto]={f1[0]:.3f}, {f1[1]:.3f} | ROC-AUC={auc:.3f}")

# ==============================
# 3) Nuevos 50 (CSV) + descarga
# ==============================
with st.expander("🟢 Nuevos 50 (CSV)", expanded=False):
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
        up = st.file_uploader("Sube CSV separado por comas", type=["csv"])
        if up is not None:
            try:
                nuevo = pd.read_csv(up, sep=",", encoding="utf-8")
            except Exception:
                up.seek(0); nuevo = pd.read_csv(up, sep=",", encoding="latin-1")

    if nuevo is not None:
        # mapear columnas del nuevo dataset usando FEATURES ya definidas
        for c in FEATURES:
            nuevo[c] = pd.to_numeric(nuevo.get(c, np.nan), errors="coerce")
        df50 = nuevo[FEATURES].head(50).dropna().copy()
        st.write(f"Filas válidas entre los primeros 50: **{len(df50)}**")
        st.dataframe(df50.head(10), use_container_width=True)

        # Modelo/Scaler de regresión (de sesión o disco)
        reg_model = st.session_state.get("reg_model", None)
        reg_scaler = st.session_state.get("reg_scaler", None)
        if reg_model is None or reg_scaler is None:
            try:
                import tensorflow as tf  # import on demand
                if os.path.exists(MODEL_PATH_H5) and os.path.exists(SCALER_REG_PATH):
                    reg_model = tf.keras.models.load_model(MODEL_PATH_H5)
                    reg_scaler = joblib.load(SCALER_REG_PATH)
            except Exception as e:
                st.info("No hay modelo/scaler guardados todavía. Entrena en la pestaña azul.")
                st.exception(e)

        out = df50.copy()

        if reg_model is not None and reg_scaler is not None:
            X50 = reg_scaler.transform(df50.values)
            y50_log = reg_model.predict(X50)
            chl50 = np.expm1(y50_log).ravel()
            chl50 = np.clip(chl50, 0, None)
            out["Clorofila predicha (µg/L)"] = chl50

            gA, gB, gC = st.columns(3)
            with gA:
                fig, ax = fig_small()
                ax.plot(range(len(chl50)), chl50)
                ax.set_title('Serie clorofila (50)')
                ax.set_xlabel('Índice'); ax.set_ylabel('µg/L')
                st.pyplot(fig)
            with gB:
                fig, ax = fig_small()
                ax.hist(chl50, bins=15)
                ax.set_title('Hist clorofila (50)')
                ax.set_xlabel('µg/L'); ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
            with gC:
                fig, ax = fig_small()
                ax.scatter(out[FEATURES[3]], out["Clorofila predicha (µg/L)"], alpha=0.75)
                ax.set_title(f'{FEATURES[3]} vs clorofila')
                ax.set_xlabel(FEATURES[3]); ax.set_ylabel('Clorofila (µg/L)')
                st.pyplot(fig)

            out["Clase por regresión (>40)"] = np.where(chl50 > UMBRAL, "Alto (>40)", "Bajo (<40)")

        # Clasificadores si están entrenados
        if "clf_scaler" in st.session_state and "clf_svm" in st.session_state and "clf_knn" in st.session_state:
            X50c = st.session_state["clf_scaler"].transform(df50.values)
            out["Clase SVM"] = np.where(st.session_state["clf_svm"].predict(X50c)==1, "Alto (>40)", "Bajo (<40)")
            out["Clase KNN"] = np.where(st.session_state["clf_knn"].predict(X50c)==1, "Alto (>40)", "Bajo (<40)")
        else:
            st.info("Entrena los clasificadores en la pestaña naranja para etiquetar SVM/KNN.")

        st.subheader("Resultados (primeros 50)")
        st.dataframe(out, use_container_width=True)

        # Descargar
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 Descargar resultados (CSV)", data=csv_bytes,
                           file_name="predicciones_dataframe_clf.csv", mime="text/csv")

# ==============================
# Requisitos sugeridos
# ==============================
st.caption("requirements.txt: streamlit, pandas, numpy, scikit-learn, matplotlib, requests, joblib, tensorflow-cpu")
