# ============================================
# Streamlit: CEA — Clorofila & Ficocianina
# - Entrena con DATOS CEA.csv
# - Predice Clorofila y Ficocianina para el estanque
# - Matrices de confusión DIFUSAS:
#     * Clorofila: 4 clases (0–2, 2–7, 7–40, ≥40)
#     * Ficocianina: 3 clases (Bajo <30, Moderado 30–90, Alto ≥90) — ajustables en la UI
# ============================================

import os, io, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ====== Opcional (para curvas de entrenamiento con red simple) ======
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    KERAS_OK = True
except Exception:
    KERAS_OK = False

# --- Estado global: regresores para predicción continua ---
TRAIN_SCALER = None
TRAIN_MODEL_CHL = None
TRAIN_MODEL_PCY = None
TRAIN_Y_LOG1P_CHL = False
TRAIN_Y_LOG1P_PCY = False

# Para exportar CSV de predicciones
if "df_pred_export" not in st.session_state:
    st.session_state.df_pred_export = None

# ------------------------- Config UI -------------------------
st.set_page_config(page_title="CEA — Clorofila & Ficocianina", layout="wide")
st.title("🧪 CEA — Entrenamiento y Predicción (Clorofila & Ficocianina)")
st.caption("Se entrena con **DATOS CEA.csv**. Matrices **difusas** para Clorofila (4 clases) y Ficocianina (3 clases) y predicción sobre datos del estanque.")

# ------------------------- Utilidades -------------------------
REQ_FEATURES = [
    "pH",
    "Temperatura (°C)",
    "Conductividad (μS/cm)",
    "Oxígeno Disuelto (mg/L)",
    "Turbidez (NTU)",
]
TARGET_CHL = "Clorofila (μg/L)"
TARGET_PCY = "Ficocianina (μg/L)"  # ajusta alias si tu archivo usa otro

# Clases para CLOROFILA (4)
BINS_CHL = [0, 2, 7, 40, np.inf]
LABELS_CHL = ["Muy bajo (0–2)", "Bajo (2–7)", "Moderado (7–40)", "Muy alto (≥40)"]

# Clases para FICOCIANINA (3) — umbrales ajustables en UI (por defecto 30 y 90)
DEFAULT_PCY_CUT1 = 30.0
DEFAULT_PCY_CUT2 = 90.0
LABELS_PCY = ["Bajo riesgo (<30)", "Riesgo moderado (30–90)", "Alto riesgo (≥90)"]

# ------------------------- Normalización flexible -------------------------
_def_map = {
    # entradas
    "ph": "pH",
    "temperatura": "Temperatura (°C)",
    "temperatura (c)": "Temperatura (°C)",
    "temp (°c)": "Temperatura (°C)",
    "temp": "Temperatura (°C)",

    # conductividad
    "conductividad (us/cm)": "Conductividad (μS/cm)",
    "conductividad(us/cm)": "Conductividad (μS/cm)",
    "conductividad (s/cm)": "Conductividad (μS/cm)",
    "conductividad": "Conductividad (μS/cm)",

    # oxígeno (variantes frecuentes)
    "oxígeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
    "oxigeno disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",
    "oxígeno disuelto (mgl)": "Oxígeno Disuelto (mg/L)",
    "oxigeno disuelto (mgl)": "Oxígeno Disuelto (mg/L)",
    "oxygen dissolved (mg/l)": "Oxígeno Disuelto (mg/L)",
    "dissolved oxygen (mg/l)": "Oxígeno Disuelto (mg/L)",
    "do (mg/l)": "Oxígeno Disuelto (mg/L)",
    "od (mg/l)": "Oxígeno Disuelto (mg/L)",
    "o2 disuelto (mg/l)": "Oxígeno Disuelto (mg/L)",

    # objetivos
    "clorofila (μg/l)": TARGET_CHL,
    "clorofila (ug/l)": TARGET_CHL,
    "clorofila": TARGET_CHL,
    "chlorophyll a": TARGET_CHL,
    "chlorophyll-a": TARGET_CHL,

    "ficocianina (μg/l)": TARGET_PCY,
    "ficocianina (ug/l)": TARGET_PCY,
    "ficocianina": TARGET_PCY,
    "phycocyanin (μg/l)": TARGET_PCY,
    "phycocyanin (ug/l)": TARGET_PCY,
    "phycocyanin": TARGET_PCY,
}

def _strip_accents(text: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', text) if not unicodedata.combining(ch))

def _canon(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("µ", "u").replace("μ", "u")
    s = "".join(ch for ch in s if ch.isprintable())
    s = _strip_accents(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = _canon(c)
        if k in _def_map:
            ren[c] = _def_map[k]; continue
        if "conductividad" in k and ("us/cm" in k or "uscm" in k or "s/cm" in k):
            ren[c] = "Conductividad (μS/cm)"; continue
        if "temperatura" in k or k.startswith("temp"):
            ren[c] = "Temperatura (°C)"; continue
        if "turbidez" in k or "turbiedad" in k or "ntu" in k:
            ren[c] = "Turbidez (NTU)"; continue
        if (("oxigeno" in k or "oxygen" in k or "dissolved oxygen" in k
             or re.search(r"\bdo\b", k) or re.search(r"\bod\b", k) or "o2" in k)
            and ("mg/l" in k or "mg l" in k or "mg" in k)):
            ren[c] = "Oxígeno Disuelto (mg/L)"; continue
        if "clorofila" in k or "chlorophyll" in k:
            ren[c] = TARGET_CHL; continue
        if "ficocianin" in k or "phycocyanin" in k:
            ren[c] = TARGET_PCY; continue
        ren[c] = c
    out = df.rename(columns=ren)
    out.columns = [col.replace("(mg/l)", "(mg/L)") for col in out.columns]
    return out

def read_csv_robust(path_or_buffer) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(path_or_buffer, encoding=enc, sep=sep)
            except Exception:
                continue
    return pd.read_csv(path_or_buffer, engine="python")

def to_numeric_smart(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\u00A0", "", regex=False).str.strip()
    def fix(x: str) -> str:
        x = x.replace(" ", "")
        if "," in x and "." in x:
            x = x.replace(",", "")
        elif "," in x and "." not in x:
            x = x.replace(",", ".")
        return x
    s = s.apply(fix)
    return pd.to_numeric(s, errors="coerce")

# ---------- Fuzzy helpers ----------
def _trapezoid(x, a, b, c, d):
    x = float(x)
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    if a < x < b: return (x - a) / (b - a) if b > a else 1.0
    if c < x < d: return (d - x) / (d - c) if d > c else 1.0
    return 0.0

def _right_shoulder(x, a, b):
    x = float(x)
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x - a) / (b - a) if b > a else 1.0

# 4 clases (Clorofila)
def memberships_4classes(x, cut1, cut2, cut3, eps1, eps2, eps3):
    m0 = _trapezoid(x, 0.0, 0.0, cut1 - eps1, cut1 + eps1)
    m1 = _trapezoid(x, cut1 - eps1, cut1 + eps1, cut2 - eps2, cut2 + eps2)
    m2 = _trapezoid(x, cut2 - eps2, cut2 + eps2, cut3 - eps3, cut3 + eps3)
    m3 = _right_shoulder(x, cut3 - eps3, cut3 + eps3)
    v = np.array([m0, m1, m2, m3], dtype=float)
    s = v.sum()
    return v / s if s > 0 else v

def fuzzy_confusion_from_probs_4(y_true_values, pred_proba, cut1, cut2, cut3, eps1, eps2, eps3):
    M = np.zeros((4, 4), dtype=float)
    for t, q in zip(y_true_values, pred_proba):
        mu_t = memberships_4classes(float(t), cut1, cut2, cut3, eps1, eps2, eps3)
        q = np.asarray(q, dtype=float)
        q = q / q.sum() if q.sum() > 0 else q
        M += np.outer(mu_t, q)
    return M

# 3 clases (Ficocianina)
def memberships_3classes(x, cut1, cut2, eps1, eps2):
    # 0: bajo (0..cut1)
    m0 = _trapezoid(x, 0.0, 0.0, cut1 - eps1, cut1 + eps1)
    # 1: moderado (cut1..cut2)
    m1 = _trapezoid(x, cut1 - eps1, cut1 + eps1, cut2 - eps2, cut2 + eps2)
    # 2: alto (>=cut2)
    m2 = _right_shoulder(x, cut2 - eps2, cut2 + eps2)
    v = np.array([m0, m1, m2], dtype=float)
    s = v.sum()
    return v / s if s > 0 else v

def fuzzy_confusion_from_probs_3(y_true_values, pred_proba, cut1, cut2, eps1, eps2):
    M = np.zeros((3, 3), dtype=float)
    for t, q in zip(y_true_values, pred_proba):
        mu_t = memberships_3classes(float(t), cut1, cut2, eps1, eps2)
        q = np.asarray(q, dtype=float)
        q = q / q.sum() if q.sum() > 0 else q
        M += np.outer(mu_t, q)
    return M

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

def align_proba_to_labels(proba: np.ndarray, classes_pred, labels_order):
    classes_pred = list(classes_pred) if classes_pred is not None else []
    idx_map = {c: i for i, c in enumerate(classes_pred)}
    n = proba.shape[0]; k = len(labels_order)
    out = np.zeros((n, k), dtype=float)
    for j, lab in enumerate(labels_order):
        if lab in idx_map:
            out[:, j] = proba[:, idx_map[lab]]
        else:
            out[:, j] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, out=out, where=row_sums > 0)
    return out

# ------------------------- 1) Tabla CEA -------------------------
DEFAULT_DIR_CEA = Path("datasets_lagos")
DEFAULT_DIR_POND = Path("pruebas_piloto")
DEFAULT_CEA = DEFAULT_DIR_CEA / "DATOS CEA.csv"

st.subheader("📄 Datos CEA — vista inicial")

cea_path_input = st.text_input("Ruta a **DATOS CEA.csv**", value=str(DEFAULT_CEA))
cea_path = Path(cea_path_input)

if not cea_path.exists():
    st.warning("No encuentro el archivo en la ruta indicada. Puedes **subirlo** aquí abajo.")
    up = st.file_uploader("Sube DATOS CEA.csv", type=["csv"])
    if up is None:
        st.stop()
    df_cea = read_csv_robust(up)
else:
    df_cea = read_csv_robust(cea_path)

# Normaliza encabezados
df_cea = normalize_columns(df_cea)

# Renombrado silencioso de Oxígeno Disuelto si no está canónico
if "Oxígeno Disuelto (mg/L)" not in df_cea.columns:
    cands = []
    for col in df_cea.columns:
        kc = _canon(col)
        if (("oxigeno" in kc or "oxygen" in kc or "dissolved oxygen" in kc
             or re.search(r"\bdo\b", kc) or re.search(r"\bod\b", kc) or "o2" in kc)
            and ("mg/l" in kc or "mg l" in kc or "mg" in kc)):
            cands.append(col)
    if cands:
        df_cea.rename(columns={cands[0]: "Oxígeno Disuelto (mg/L)"}, inplace=True)

# Vista previa
st.markdown("**Vista previa (10 primeras filas):**")
st.table(df_cea.head(10))
with st.expander("⬇️ Ver todos los datos CEA"):
    st.dataframe(df_cea, use_container_width=True)

# Conversión numérica robusta
for c in REQ_FEATURES + [TARGET_CHL, TARGET_PCY]:
    if c in df_cea.columns:
        df_cea[c] = to_numeric_smart(df_cea[c])

# Validación de X
faltantes_x = [c for c in REQ_FEATURES if c not in df_cea.columns]
if faltantes_x:
    st.error(f"Faltan columnas de entrada en CEA: {faltantes_x}. Ajusta nombres o el mapa en el script.")
    st.stop()

# Al menos un target
targets_present = [t for t in [TARGET_CHL, TARGET_PCY] if t in df_cea.columns]
if not targets_present:
    st.error(f"No se encontró ninguna columna objetivo ({TARGET_CHL} o {TARGET_PCY}) en CEA.")
    st.stop()

# Limpieza mínima
base_x = df_cea.dropna(subset=REQ_FEATURES).reset_index(drop=True)
X_all = base_x[REQ_FEATURES].values

# ------------------------- 2) Curvas de entrenamiento y modelos -------------------------
st.subheader("📈 Curva(s) de entrenamiento y nota")
col_curve, col_note = st.columns([2, 1])

with col_curve:
    if not KERAS_OK:
        st.warning("TensorFlow/Keras no está disponible. Se mostrará una curva **simulada**.")
        losses = np.linspace(1.0, 0.25, 60) + 0.05*np.random.randn(60)
        val_losses = losses + 0.05*np.random.randn(60) + 0.05
        fig_loss, ax = plt.subplots()
        ax.plot(losses, label="Pérdida entrenamiento")
        ax.plot(val_losses, label="Pérdida validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax.set_title("Curva de entrenamiento (simulada)")
        ax.grid(True); ax.legend(); fig_loss.tight_layout()
        st.pyplot(fig_loss, use_container_width=True)

        TRAIN_SCALER = None
        TRAIN_MODEL_CHL = None
        TRAIN_MODEL_PCY = None
        TRAIN_Y_LOG1P_CHL = False
        TRAIN_Y_LOG1P_PCY = False
    else:
        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)
        TRAIN_SCALER = scaler

        def build_regressor(input_dim: int):
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.15),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=keras.losses.Huber(delta=1.0)
            )
            return model

        fig_loss = None

        # ---- Modelo Clorofila
        if TARGET_CHL in df_cea.columns:
            base_chl = df_cea.dropna(subset=REQ_FEATURES + [TARGET_CHL]).reset_index(drop=True)
            if not base_chl.empty:
                Xc = scaler.transform(base_chl[REQ_FEATURES].values)
                yc = base_chl[TARGET_CHL].to_numpy()
                X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
                TRAIN_Y_LOG1P_CHL = True
                y_tr_t = np.log1p(y_tr) if TRAIN_Y_LOG1P_CHL else y_tr
                y_te_t = np.log1p(y_te) if TRAIN_Y_LOG1P_CHL else y_te

                model_chl = build_regressor(X_tr.shape[1])
                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=0)
                rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=0)
                hist = model_chl.fit(X_tr, y_tr_t, validation_data=(X_te, y_te_t),
                                     epochs=400, batch_size=32, verbose=0, callbacks=[es, rl])
                TRAIN_MODEL_CHL = model_chl

                fig_loss, ax = plt.subplots()
                ax.plot(hist.history["loss"], label="Pérdida entrenamiento (Clorofila)")
                ax.plot(hist.history["val_loss"], label="Pérdida validación (Clorofila)")
                ax.set_xlabel("Época"); ax.set_ylabel("Loss")
                ax.set_title("Curva de entrenamiento (NN Clorofila — CEA)")
                ax.grid(True); ax.legend(); fig_loss.tight_layout()

        # ---- Modelo Ficocianina
        if TARGET_PCY in df_cea.columns:
            base_pcy = df_cea.dropna(subset=REQ_FEATURES + [TARGET_PCY]).reset_index(drop=True)
            if not base_pcy.empty:
                Xp = scaler.transform(base_pcy[REQ_FEATURES].values)
                yp = base_pcy[TARGET_PCY].to_numpy()
                X_tr, X_te, y_tr, y_te = train_test_split(Xp, yp, test_size=0.2, random_state=42)
                TRAIN_Y_LOG1P_PCY = True
                y_tr_t = np.log1p(y_tr) if TRAIN_Y_LOG1P_PCY else y_tr
                y_te_t = np.log1p(y_te) if TRAIN_Y_LOG1P_PCY else y_te

                model_pcy = build_regressor(X_tr.shape[1])
                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=0)
                rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=0)
                hist_p = model_pcy.fit(X_tr, y_tr_t, validation_data=(X_te, y_te_t),
                                       epochs=400, batch_size=32, verbose=0, callbacks=[es, rl])
                TRAIN_MODEL_PCY = model_pcy

                if fig_loss is None:
                    fig_loss, ax = plt.subplots()
                    ax.plot(hist_p.history["loss"], label="Pérdida entrenamiento (Ficocianina)")
                    ax.plot(hist_p.history["val_loss"], label="Pérdida validación (Ficocianina)")
                    ax.set_xlabel("Época"); ax.set_ylabel("Loss")
                    ax.set_title("Curva de entrenamiento (NN Ficocianina — CEA)")
                    ax.grid(True); ax.legend(); fig_loss.tight_layout()

        if fig_loss is not None:
            st.pyplot(fig_loss, use_container_width=True)
        else:
            st.info("No hay suficientes filas válidas para dibujar curvas de entrenamiento.")

with col_note:
    st.info(
        """
        **Nota:** Se entrenan dos regresores (si hay datos): **Clorofila** y **Ficocianina**
        usando pH, temperatura, conductividad, oxígeno disuelto y turbidez.
        Matrices difusas:
        - Clorofila: 4 clases fijas (2, 7, 40 μg/L).
        - Ficocianina: 3 clases (por defecto 30 y 90 μg/L), ajustables.
        """
    )

# ------------------------- 3) Matrices difusas con CEA — CLOROFILA (4 clases) -------------------------
st.subheader("🧩 Matrices de confusión **difusas** — Clorofila (CEA)")
base_cls_chl = df_cea.dropna(subset=REQ_FEATURES + [TARGET_CHL]).reset_index(drop=True)

if not base_cls_chl.empty:
    # Datos y limpieza robusta
    X_all_cls = base_cls_chl[REQ_FEATURES].to_numpy()
    y_all_num = pd.to_numeric(base_cls_chl[TARGET_CHL], errors="coerce").to_numpy()
    finite_mask = np.isfinite(y_all_num)
    X_all_cls = X_all_cls[finite_mask]
    y_all_num = np.clip(y_all_num[finite_mask], 0.0, None)

    X_train, X_test, y_train_num, y_test_num = train_test_split(X_all_cls, y_all_num, test_size=0.20, random_state=42)
    y_train_cls = pd.cut(y_train_num, bins=BINS_CHL, labels=LABELS_CHL, right=False, include_lowest=True)

    # Sincroniza y valida clases (FIX: sin .to_numpy())
    mask_ok = ~np.asarray(pd.isna(y_train_cls))
    X_train, y_train_num, y_train_cls = X_train[mask_ok], y_train_num[mask_ok], y_train_cls[mask_ok]
    if pd.Series(y_train_cls).nunique() < 2:
        st.error("No hay variedad de clases suficiente para entrenar el clasificador de **Clorofila**.")
        st.stop()

    svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale",
                                                 class_weight="balanced", probability=True, random_state=42))
    knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))

    svm_clf.fit(X_train, y_train_cls)
    knn_clf.fit(X_train, y_train_cls)

    proba_svm = svm_clf.predict_proba(X_test)
    proba_knn = knn_clf.predict_proba(X_test)

    svm_classes = svm_clf.named_steps[list(svm_clf.named_steps.keys())[-1]].classes_
    knn_classes = knn_clf.named_steps[list(knn_clf.named_steps.keys())[-1]].classes_

    proba_svm_al = align_proba_to_labels(proba_svm, svm_classes, LABELS_CHL)
    proba_knn_al = align_proba_to_labels(proba_knn, knn_classes, LABELS_CHL)

    cm_svm_fuzzy = fuzzy_confusion_from_probs_4(y_test_num, proba_svm_al, 2.0, 7.0, 40.0, 0.3, 1.0, 5.0)
    cm_knn_fuzzy = fuzzy_confusion_from_probs_4(y_test_num, proba_knn_al, 2.0, 7.0, 40.0, 0.3, 1.0, 5.0)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_confusion_matrix_pretty_float(cm_svm_fuzzy, LABELS_CHL, "Matriz **difusa** — SVM (Clorofila — CEA)"),
                  use_container_width=True)
        st.caption(f"Suma de pesos (SVM): {cm_svm_fuzzy.sum():.2f}")
    with c2:
        st.pyplot(plot_confusion_matrix_pretty_float(cm_knn_fuzzy, LABELS_CHL, "Matriz **difusa** — KNN (Clorofila — CEA)"),
                  use_container_width=True)
        st.caption(f"Suma de pesos (KNN): {cm_knn_fuzzy.sum():.2f}")

    st.info("Las clases difusas para **Clorofila** usan umbrales fijos: 2, 7 y 40 μg/L.")
else:
    st.warning("No hay datos suficientes de **Clorofila** para matrices difusas.")

# ------------------------- 4) Matrices difusas con CEA — FICOCIANINA (3 clases) -------------------------
st.subheader("🧩 Matrices de confusión **difusas** — Ficocianina (CEA)")

# Controles de umbrales y suavidad para Ficocianina (3 clases)
with st.expander("⚙️ Umbrales y suavidad (Ficocianina)"):
    col_a, col_b = st.columns(2)
    cut1 = col_a.number_input("Umbral Bajo→Moderado (μg/L)", min_value=0.0, value=float(DEFAULT_PCY_CUT1), step=1.0)
    cut2 = col_b.number_input("Umbral Moderado→Alto (μg/L)", min_value=float(max(cut1 + 0.1, 1.0)), value=float(DEFAULT_PCY_CUT2), step=1.0)

    col_e1, col_e2 = st.columns(2)
    eps1 = col_e1.number_input("ε alrededor del umbral 1", min_value=0.0, value=3.0, step=0.5)
    eps2 = col_e2.number_input("ε alrededor del umbral 2", min_value=0.0, value=10.0, step=0.5)

base_cls_pcy = df_cea.dropna(subset=REQ_FEATURES + [TARGET_PCY]).reset_index(drop=True)

if not base_cls_pcy.empty:
    # Datos y limpieza robusta
    X_all_pcy = base_cls_pcy[REQ_FEATURES].to_numpy()
    y_all_pcy_num = pd.to_numeric(base_cls_pcy[TARGET_PCY], errors="coerce").to_numpy()
    finite_mask_p = np.isfinite(y_all_pcy_num)
    X_all_pcy = X_all_pcy[finite_mask_p]
    y_all_pcy_num = np.clip(y_all_pcy_num[finite_mask_p], 0.0, None)

    # Bins 3 clases con tus cortes
    bins_pcy = [0.0, float(cut1), float(cut2), np.inf]

    X_train_p, X_test_p, y_train_num_p, y_test_num_p = train_test_split(
        X_all_pcy, y_all_pcy_num, test_size=0.20, random_state=42
    )
    y_train_cls_pcy = pd.cut(y_train_num_p, bins=bins_pcy, labels=LABELS_PCY, right=False, include_lowest=True)

    # Sincroniza y valida (FIX: sin .to_numpy())
    mask_ok_p = ~np.asarray(pd.isna(y_train_cls_pcy))
    X_train_p, y_train_num_p, y_train_cls_pcy = X_train_p[mask_ok_p], y_train_num_p[mask_ok_p], y_train_cls_pcy[mask_ok_p]
    if pd.Series(y_train_cls_pcy).nunique() < 2:
        st.error("No hay variedad de clases suficiente para entrenar el clasificador de **Ficocianina**. Ajusta umbrales o revisa datos.")
        st.stop()

    svm_clf_pcy = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale",
                                                     class_weight="balanced", probability=True, random_state=42))
    knn_clf_pcy = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))

    svm_clf_pcy.fit(X_train_p, y_train_cls_pcy)
    knn_clf_pcy.fit(X_train_p, y_train_cls_pcy)

    proba_svm_pcy = svm_clf_pcy.predict_proba(X_test_p)
    proba_knn_pcy = knn_clf_pcy.predict_proba(X_test_p)

    svm_classes_pcy = svm_clf_pcy.named_steps[list(svm_clf_pcy.named_steps.keys())[-1]].classes_
    knn_classes_pcy = knn_clf_pcy.named_steps[list(knn_clf_pcy.named_steps.keys())[-1]].classes_

    proba_svm_pcy_al = align_proba_to_labels(proba_svm_pcy, svm_classes_pcy, LABELS_PCY)
    proba_knn_pcy_al = align_proba_to_labels(proba_knn_pcy, knn_classes_pcy, LABELS_PCY)

    cm_svm_fuzzy_pcy = fuzzy_confusion_from_probs_3(y_test_num_p, proba_svm_pcy_al, float(cut1), float(cut2), float(eps1), float(eps2))
    cm_knn_fuzzy_pcy = fuzzy_confusion_from_probs_3(y_test_num_p, proba_knn_pcy_al, float(cut1), float(cut2), float(eps1), float(eps2))

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(
            plot_confusion_matrix_pretty_float(
                cm_svm_fuzzy_pcy, LABELS_PCY,
                f"Matriz **difusa** — SVM (Ficocianina — CEA)\nUmbrales: {cut1:.0f}, {cut2:.0f} μg/L"
            ),
            use_container_width=True
        )
        st.caption(f"Suma de pesos (SVM): {cm_svm_fuzzy_pcy.sum():.2f}")
    with c4:
        st.pyplot(
            plot_confusion_matrix_pretty_float(
                cm_knn_fuzzy_pcy, LABELS_PCY,
                f"Matriz **difusa** — KNN (Ficocianina — CEA)\nUmbrales: {cut1:.0f}, {cut2:.0f} μg/L"
            ),
            use_container_width=True
        )
        st.caption(f"Suma de pesos (KNN): {cm_knn_fuzzy_pcy.sum():.2f}")

    st.info(
        "Clases **Ficocianina**:\n"
        f"- **Bajo riesgo**: < {cut1:.0f} μg/L (menor probabilidad de floración visible)\n"
        f"- **Riesgo moderado**: {cut1:.0f}–{cut2:.0f} μg/L (monitoreo cercano)\n"
        f"- **Alto riesgo**: ≥ {cut2:.0f} μg/L (alta probabilidad de floración)"
    )
else:
    st.warning("No hay datos suficientes de **Ficocianina** para matrices difusas.")

# ------------------------- 5) Predicción (Clorofila y Ficocianina) con datos del estanque -------------------------
st.subheader("🧪 Predicción (Clorofila y Ficocianina) con datos del estanque")

clicked = st.button("🔮 Predecir con datos del estanque")
if clicked:
    DEFAULT_POND = DEFAULT_DIR_POND / "dataframe1.csv"
    pond_path_input = st.text_input("Ruta a **dataframe del estanque**", value=str(DEFAULT_POND), key="pond_path")
    pond_path = Path(pond_path_input)

    if not pond_path.exists():
        st.warning("No encuentro **dataframe del estanque** en la ruta indicada. Sube el archivo:")
        up2 = st.file_uploader("Sube CSV del estanque", type=["csv"], key="pond")
        if up2 is None:
            st.stop()
        df_pond = read_csv_robust(up2)
    else:
        df_pond = read_csv_robust(pond_path)

    df_pond = normalize_columns(df_pond)
    for c in REQ_FEATURES + [TARGET_CHL, TARGET_PCY]:
        if c in df_pond.columns:
            df_pond[c] = to_numeric_smart(df_pond[c])

    df_pond = df_pond.dropna(subset=REQ_FEATURES).reset_index(drop=True)
    if df_pond.empty:
        st.error("El archivo del estanque no tiene filas válidas tras limpieza.")
        st.stop()

    Xp = df_pond[REQ_FEATURES].values

    # --- Predicción Clorofila (continua)
    yhat_chl = None
    if TRAIN_SCALER is not None and TRAIN_MODEL_CHL is not None:
        Xp_s = TRAIN_SCALER.transform(Xp)
        y_pred_t = TRAIN_MODEL_CHL.predict(Xp_s, verbose=0).ravel()
        yhat_chl = np.expm1(y_pred_t) if TRAIN_Y_LOG1P_CHL else y_pred_t
    else:
        # Fallback con SVM de Clorofila si se entrenó
        if 'svm_clf' in locals():
            proba_svm_p = svm_clf.predict_proba(Xp)
            proba_svm_p_al = align_proba_to_labels(proba_svm_p, svm_classes, LABELS_CHL)
            centers = np.array([1.0, 4.5, 20.0, 60.0])
            yhat_chl = proba_svm_p_al @ centers
        else:
            yhat_chl = np.full(len(Xp), np.nan)
    yhat_chl = np.clip(yhat_chl, 0.0, None) if yhat_chl is not None else None

    # --- Predicción Ficocianina (continua)
    yhat_pcy = None
    if TRAIN_SCALER is not None and TRAIN_MODEL_PCY is not None:
        Xp_s = TRAIN_SCALER.transform(Xp)
        y_pred_t = TRAIN_MODEL_PCY.predict(Xp_s, verbose=0).ravel()
        yhat_pcy = np.expm1(y_pred_t) if TRAIN_Y_LOG1P_PCY else y_pred_t
        yhat_pcy = np.clip(yhat_pcy, 0.0, None)
    else:
        yhat_pcy = np.full(len(Xp), np.nan)

    # --- Exportar
    df_pred_export = df_pond.copy()
    df_pred_export["Clorofila_predicha (μg/L)"] = yhat_chl if yhat_chl is not None else np.nan
    df_pred_export["Ficocianina_predicha (μg/L)"] = yhat_pcy

    st.session_state.df_pred_export = df_pred_export
    st.success("✅ Predicciones generadas para el estanque (Clorofila y Ficocianina). Revisa y descarga abajo.")

# ========= Botón inferior: Descargar CSV =========
col_right = st.columns(2)[1]
with col_right:
    df_pred = st.session_state.get("df_pred_export")
    if isinstance(df_pred, pd.DataFrame) and not df_pred.empty:
        st.download_button(
            "⬇️ Descargar predicciones (.csv)",
            data=df_pred.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_estanque_CEA.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.download_button(
            "⬇️ Descargar predicciones (.csv)",
            data=b"",
            disabled=True,
            help="Primero presiona 'Predecir con datos del estanque'.",
            use_container_width=True,
        )
