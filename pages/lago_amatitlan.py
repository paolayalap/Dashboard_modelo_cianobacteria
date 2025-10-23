# ============================================
# Streamlit: Visualización AMSA + Entrenamiento + Fuzzy Confusion (SVM/KNN)
# Requisitos: streamlit, numpy, pandas, scikit-learn, matplotlib, tensorflow/keras (opcional, para curva)
# Ejecuta:  streamlit run streamlit_app_amsa.py
# ============================================

import os
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ====== Opcional (para curva de entrenamiento con red simple) ======
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    KERAS_OK = True
except Exception:
    KERAS_OK = False

# ------------------------- Config UI -------------------------
st.set_page_config(page_title="AMSA — Tabla, Curva y Matrices Fuzzy", layout="wide")
st.title("📊 AMSA — Tabla, Curva de Entrenamiento y Matrices de Confusión Difusas")
st.caption("Se entrena un modelo con **DATOS AMSA.csv**. Luego se muestran matrices difusas para SVM y KNN. Finalmente, puedes evaluar con `dataframe.csv` del estanque.")

# ------------------------- Utilidades -------------------------
REQ_FEATURES = [
    "pH",
    "Temperatura (°C)",
    "Conductividad (μS/cm)",
    "Oxígeno Disuelto (mg/L)",
    "Turbidez (NTU)",
]
TARGET = "Clorofila (μg/L)"
BINS = [0, 2, 7, 40, np.inf]
LABELS = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]

# Normalización flexible de nombres de columnas
_def_map = {
    "ph": "pH",
    "temperatura": "Temperatura (°C)",
    "temperatura (c)": "Temperatura (°C)",
    "temp (°c)": "Temperatura (°C)",
    "temp": "Temperatura (°C)",
    "conductividad (us/cm)": "Conductividad (μS/cm)",
    "conductividad(us/cm)": "Conductividad (μS/cm)",
    "conductividad (s/cm)": "Conductividad (μS/cm)",
    "conductividad": "Conductividad (μS/cm)",
    "oxigeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
    "oxígeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
    "oxigeno disuelto": "Oxígeno Disuelto (mg/L)",
    "do (mg/l)": "Oxígeno Disuelto (mg/L)",
    "turbidez (ntu)": "Turbidez (NTU)",
    "turbiedad (ntu)": "Turbidez (NTU)",
    "turbiedad": "Turbidez (NTU)",
    "turbidez": "Turbidez (NTU)",
    "clorofila (ug/l)": TARGET,
    "clorofila (μg/l)": TARGET,
    "clorofila": TARGET,
}

def _canon(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("µ", "u").replace("μ", "u").replace("?", "u")
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = _canon(c)
        if k in _def_map:
            ren[c] = _def_map[k]
            continue
        if "conductividad" in k and ("us/cm" in k or "s/cm" in k):
            ren[c] = "Conductividad (μS/cm)"
        elif ("oxigeno" in k or "oxígeno" in k) and "mg/l" in k:
            ren[c] = "Oxígeno Disuelto (mg/L)"
        elif "temperatura" in k or k.startswith("temp"):
            ren[c] = "Temperatura (°C)"
        else:
            ren[c] = c
    return df.rename(columns=ren)

# Lectura robusta de CSV
ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
SEPS = [",", ";", "\t", "|"]

def read_csv_robust(path_or_buffer) -> pd.DataFrame:
    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                return pd.read_csv(path_or_buffer, encoding=enc, sep=sep)
            except Exception:
                continue
    # último intento con motor python autodetección
    try:
        return pd.read_csv(path_or_buffer, engine="python")
    except Exception as e:
        raise e

# Fuzzy helpers

def _trapezoid(x, a, b, c, d):
    x = float(x)
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return 1.0 if b == a else (x - a) / (b - a)
    if c < x < d:
        return 1.0 if d == c else (d - x) / (d - c)
    return 0.0

def _right_shoulder(x, a, b):
    x = float(x)
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a) if b > a else 1.0

# eps: anchos de suavizado por umbral (2, 7, 40 µg/L)
DEFAULT_EPS = (0.3, 1.0, 5.0)

def fuzzy_memberships_scalar(x, eps=DEFAULT_EPS):
    e1, e2, e3 = eps
    m0 = _trapezoid(x, 0.0, 0.0, 2.0 - e1, 2.0 + e1)           # 0–2
    m1 = _trapezoid(x, 2.0 - e1, 2.0 + e1, 7.0 - e2, 7.0 + e2) # 2–7
    m2 = _trapezoid(x, 7.0 - e2, 7.0 + e2, 40.0 - e3, 40.0 + e3) # 7–40
    m3 = _right_shoulder(x, 40.0 - e3, 40.0 + e3)               # ≥40
    v = np.array([m0, m1, m2, m3], dtype=float)
    s = v.sum()
    return v / s if s > 0 else v

def fuzzy_confusion_from_probs(y_true_values, pred_proba, n_classes=4, eps=DEFAULT_EPS):
    M = np.zeros((n_classes, n_classes), dtype=float)
    for t, q in zip(y_true_values, pred_proba):
        mu_t = fuzzy_memberships_scalar(t, eps)
        q = np.asarray(q, dtype=float)
        q = q / q.sum() if q.sum() > 0 else q
        M += np.outer(mu_t, q)
    return M

# Plot pretty confusion (float)
from matplotlib.patches import Rectangle

def plot_confusion_matrix_pretty_float(cm, labels, title, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    n = len(labels)
    ax.set_xlim(-0.5, n-0.5); ax.set_ylim(n-0.5, -0.5)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(n+1):
        ax.axhline(i-0.5, color="#888", lw=0.6, alpha=0.6)
        ax.axvline(i-0.5, color="#888", lw=0.6, alpha=0.6)
    for i in range(n):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, facecolor="#78c679", alpha=0.30, edgecolor="none"))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm[i, j]), va="center", ha="center", fontsize=14, fontweight="bold")
    ax.set_xlabel("Etiqueta predicha"); ax.set_ylabel("Etiqueta real (difusa)")
    fig.tight_layout()
    return fig

# Reordenar predict_proba al orden deseado

def align_proba_to_labels(proba: np.ndarray, classes_pred, labels_order):
    classes_pred = list(classes_pred) if classes_pred is not None else []
    idx_map = {c: i for i, c in enumerate(classes_pred)}
    n = proba.shape[0]
    k = len(labels_order)
    out = np.zeros((n, k), dtype=float)
    for j, lab in enumerate(labels_order):
        if lab in idx_map:
            out[:, j] = proba[:, idx_map[lab]]
        else:
            out[:, j] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, out=out, where=row_sums > 0)
    return out

# ------------------------- 1) Tabla AMSA -------------------------
DEFAULT_DIR_AMSA = Path("datasets_lagos")
DEFAULT_DIR_POND = Path("pruebas_piloto")
DEFAULT_AMSA = DEFAULT_DIR_AMSA / "DATOS AMSA.csv"  # ruta por defecto AMSA

st.subheader("📄 Datos AMSA — vista inicial")

# Entradas de ruta (puedes cambiarlas en la UI)
amsa_path_input = st.text_input("Ruta a **DATOS AMSA.csv**", value=str(DEFAULT_AMSA))
amsa_path = Path(amsa_path_input)

if not amsa_path.exists():
    st.warning("No encuentro el archivo en la ruta indicada. Puedes **subirlo** aquí abajo.")
    up = st.file_uploader("Sube DATOS AMSA.csv", type=["csv"]) 
    if up is None:
        st.stop()
    df_amsa = read_csv_robust(up)
else:
    df_amsa = read_csv_robust(amsa_path)

# Normaliza encabezados comunes
df_amsa = normalize_columns(df_amsa)

# Mostrar tabla estática (primeros 10) y expander para todo
st.markdown("**Vista previa (10 primeras filas):**")
st.table(df_amsa.head(10))
with st.expander("⬇️ Ver todos los datos AMSA"):
    st.dataframe(df_amsa, use_container_width=True)

# Conversión numérica de columnas requeridas (silenciosa)
for c in REQ_FEATURES + [TARGET]:
    if c in df_amsa.columns:
        df_amsa[c] = (
            df_amsa[c]
            .astype(str)
            .str.replace("\u00A0", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df_amsa[c] = pd.to_numeric(df_amsa[c], errors="coerce")

# Validación de columnas
faltantes = [c for c in REQ_FEATURES + [TARGET] if c not in df_amsa.columns]
if faltantes:
    st.error(f"Faltan columnas requeridas en AMSA: {faltantes}. Renombra tus columnas o ajusta el mapa en el script.")
    st.stop()

# Limpieza básica
base = df_amsa.dropna(subset=REQ_FEATURES + [TARGET]).reset_index(drop=True)
if base.empty:
    st.error("El archivo AMSA no tiene filas válidas tras limpieza.")
    st.stop()

X_all = base[REQ_FEATURES].values
y_all = base[TARGET].values

# ------------------------- 2) Curva de entrenamiento + Nota -------------------------
st.subheader("📈 Curva de entrenamiento (izquierda) y nota (derecha)")
col_curve, col_note = st.columns([2, 1])

with col_curve:
    if not KERAS_OK:
        st.warning("TensorFlow/Keras no está disponible. Se mostrará una curva ficticia basada en una regresión simple.")
        # Curva ficticia (descendente) para no fallar si no hay Keras
        losses = np.linspace(1.0, 0.2, 60) + 0.05*np.random.randn(60)
        val_losses = losses + 0.05*np.random.randn(60) + 0.05
        fig_loss, ax = plt.subplots()
        ax.plot(losses, label="Pérdida entrenamiento")
        ax.plot(val_losses, label="Pérdida validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax.set_title("Curva de entrenamiento (simulada)")
        ax.grid(True); ax.legend(); fig_loss.tight_layout()
        st.pyplot(fig_loss, use_container_width=True)
    else:
        # Pequeña red para historial de pérdida
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = keras.Sequential([
            layers.Input(shape=(X_tr_s.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        hist = model.fit(X_tr_s, y_tr, validation_split=0.2, epochs=300, batch_size=32, verbose=0)

        fig_loss, ax = plt.subplots()
        ax.plot(hist.history["loss"], label="Pérdida entrenamiento")
        ax.plot(hist.history["val_loss"], label="Pérdida validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.set_title("Curva de entrenamiento (Regresión NN sobre AMSA)")
        ax.grid(True); ax.legend(); fig_loss.tight_layout()
        st.pyplot(fig_loss, use_container_width=True)

        # Guardamos scaler y modelo para usar luego si hace falta (solo en RAM)
        TRAIN_SCALER = scaler
        TRAIN_MODEL = model

with col_note:
    st.info(
        """
        **Nota breve:** La curva muestra cómo evoluciona la *pérdida* durante el entrenamiento
        y validación del modelo de **regresión** que aprende a estimar la clorofila (μg/L)
        a partir de pH, temperatura, conductividad, oxígeno disuelto y turbidez (datos AMSA).
        Una curva descendente y estable sugiere buen ajuste sin sobreajuste.
        """
    )
    user_note = st.text_area("✍️ Puedes editar esta explicación:", value="La pérdida de validación converge sin aumentar, indicando buen generalizado.")

# ------------------------- 3) Matrices difusas (SVM y KNN) + Nota -------------------------
st.subheader("🧩 Matrices de confusión **difusas** con AMSA (SVM y KNN)")

# Preparar clases a partir de y_all (rango de clorofila)
y_bins = pd.cut(y_all, bins=BINS, labels=LABELS, right=False)

X_train, X_test, y_train_num, y_test_num = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
y_train_cls = pd.cut(y_train_num, bins=BINS, labels=LABELS, right=False)

# SVM y KNN con probas
svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=42))
knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))

svm_clf.fit(X_train, y_train_cls)
knn_clf.fit(X_train, y_train_cls)

proba_svm = svm_clf.predict_proba(X_test)
proba_knn = knn_clf.predict_proba(X_test)

# Extraer clases_ reales del último estimador en pipeline
svm_classes = svm_clf.named_steps[list(svm_clf.named_steps.keys())[-1]].classes_
knn_classes = knn_clf.named_steps[list(knn_clf.named_steps.keys())[-1]].classes_

# Reordenar columnas de proba al orden LABELS
proba_svm_al = align_proba_to_labels(proba_svm, svm_classes, LABELS)
proba_knn_al = align_proba_to_labels(proba_knn, knn_classes, LABELS)

# Matrices difusas contra valores numéricos verdaderos (y_test_num)
cm_svm_fuzzy = fuzzy_confusion_from_probs(y_test_num, proba_svm_al, n_classes=4)
cm_knn_fuzzy = fuzzy_confusion_from_probs(y_test_num, proba_knn_al, n_classes=4)

c1, c2 = st.columns(2)
with c1:
    st.pyplot(plot_confusion_matrix_pretty_float(cm_svm_fuzzy, LABELS, "Matriz **difusa** — SVM (AMSA)"), use_container_width=True)
    st.caption(f"Suma de pesos (SVM): {cm_svm_fuzzy.sum():.2f}")
with c2:
    st.pyplot(plot_confusion_matrix_pretty_float(cm_knn_fuzzy, LABELS, "Matriz **difusa** — KNN (AMSA)"), use_container_width=True)
    st.caption(f"Suma de pesos (KNN): {cm_knn_fuzzy.sum():.2f}")

st.info(
    """
    **Nota breve:** Estas matrices **difusas** consideran la cercanía a los umbrales (2, 7, 40 μg/L).
    En lugar de contar aciertos/errores duros, reparten *peso* entre clases vecinas cuando la
    clorofila real está cerca de un límite. Así, penalizan menos las confusiones razonables.
    """
)
user_note2 = st.text_area("✍️ Puedes editar esta explicación de las matrices:", value="La matriz difusa suaviza el conteo cerca de 2, 7 y 40 μg/L.")

# ------------------------- 4) Botón: Predecir con datos del estanque -------------------------
st.subheader("🧪 Predicción y matrices (difusas) con datos del estanque")

clicked = st.button("🔮 Predecir con datos del estanque")
if clicked:
    DEFAULT_POND = DEFAULT_DIR_POND / "dataframe1.csv"
pond_path_input = st.text_input("Ruta a **dataframe del estanque**", value=str(DEFAULT_POND), key="pond_path")
pond_path = Path(pond_path_input)
    if not pond_path.exists():
        st.warning("No encuentro **dataframe.csv** en el directorio actual. Sube el archivo:")
        up2 = st.file_uploader("Sube dataframe.csv", type=["csv"], key="pond")
        if up2 is None:
            st.stop()
        df_pond = read_csv_robust(up2)
    else:
        df_pond = read_csv_robust(pond_path)

    df_pond = normalize_columns(df_pond)

    # Intentamos usar la clorofila real si existe; si no, usamos proxy NN (si Keras entrenó)
    have_true = TARGET in df_pond.columns

    # Convertir features
    for c in REQ_FEATURES + ([TARGET] if have_true else []):
        if c in df_pond.columns:
            df_pond[c] = (
                df_pond[c]
                .astype(str)
                .str.replace("\u00A0", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df_pond[c] = pd.to_numeric(df_pond[c], errors="coerce")

    df_pond = df_pond.dropna(subset=REQ_FEATURES).reset_index(drop=True)
    if df_pond.empty:
        st.error("El archivo del estanque no tiene filas válidas tras limpieza.")
        st.stop()

    Xp = df_pond[REQ_FEATURES].values

    # Obtener probas SVM/KNN (entrenados con AMSA)
    proba_svm_p = svm_clf.predict_proba(Xp)
    proba_knn_p = knn_clf.predict_proba(Xp)
    proba_svm_p_al = align_proba_to_labels(proba_svm_p, svm_classes, LABELS)
    proba_knn_p_al = align_proba_to_labels(proba_knn_p, knn_classes, LABELS)

    if have_true:
        y_true_p = pd.to_numeric(df_pond[TARGET], errors="coerce").fillna(0).to_numpy()
        used_proxy = False
    else:
        # Proxy: si se entrenó la NN arriba, usarla; si no, proxy por centroide de clases predichas SVM
        if KERAS_OK and 'TRAIN_MODEL' in globals() and 'TRAIN_SCALER' in globals():
            Xp_s = TRAIN_SCALER.transform(Xp)
            y_proxy = TRAIN_MODEL.predict(Xp_s, verbose=0).ravel()
            y_true_p = np.clip(y_proxy, 0.0, None)
            used_proxy = True
        else:
            # Centroide aproximado por clase SVM (como último recurso)
            pred_cls = np.argmax(proba_svm_p_al, axis=1)
            centers = np.array([1.0, 4.5, 20.0, 60.0])  # centroides aproximados de rangos
            y_true_p = centers[pred_cls]
            used_proxy = True

    cm_svm_p = fuzzy_confusion_from_probs(y_true_p, proba_svm_p_al, n_classes=4)
    cm_knn_p = fuzzy_confusion_from_probs(y_true_p, proba_knn_p_al, n_classes=4)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.pyplot(plot_confusion_matrix_pretty_float(cm_svm_p, LABELS, "Matriz **difusa** — SVM (Estanque)"), use_container_width=True)
        st.caption(f"Suma de pesos (SVM): {cm_svm_p.sum():.2f}")
    with cc2:
        st.pyplot(plot_confusion_matrix_pretty_float(cm_knn_p, LABELS, "Matriz **difusa** — KNN (Estanque)"), use_container_width=True)
        st.caption(f"Suma de pesos (KNN): {cm_knn_p.sum():.2f}")

    if used_proxy:
        st.caption("ℹ️ En el estanque se usó **proxy** de verdad de clorofila para la matriz difusa (no había columna de clorofila real).")

    st.success("Listo. Matrices del estanque generadas.")

# ------------------------- Fin -------------------------
