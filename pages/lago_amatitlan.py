# ============================================
# Streamlit: Visualización AMSA + Entrenamiento + Fuzzy Confusion (SVM/KNN)
# Sin prompts de "Oxígeno Disuelto": auto-detección/renombrado silencioso
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

# ====== Opcional (para curva de entrenamiento con red simple) ======
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    KERAS_OK = True
except Exception:
    KERAS_OK = False

# --- Estado global para el modelo de la curva ---
TRAIN_SCALER = None
TRAIN_MODEL = None
TRAIN_Y_LOG1P = False

# ------------------------- Config UI -------------------------
st.set_page_config(page_title="AMSA — Tabla, Curva y Matrices Fuzzy", layout="wide")
st.title("📊 AMSA — Tabla, Curva de Entrenamiento y Matrices de Confusión Difusas")
st.caption("Se entrena un modelo con **DATOS AMSA.csv**. Luego se muestran matrices difusas para SVM y KNN. Finalmente, puedes evaluar con `dataframe1.csv` del estanque.")

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

    # turbidez
    "turbidez (ntu)": "Turbidez (NTU)",
    "turbiedad (ntu)": "Turbidez (NTU)",
    "turbidez": "Turbidez (NTU)",
    "turbiedad": "Turbidez (NTU)",

    # objetivo
    "clorofila (μg/l)": TARGET,
    "clorofila (ug/l)": TARGET,
    "clorofila": TARGET,
}

def _strip_accents(text: str) -> str:
    """Elimina diacríticos (tildes) para comparar de forma robusta."""
    return ''.join(ch for ch in unicodedata.normalize('NFD', text) if not unicodedata.combining(ch))

def _canon(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("µ", "u").replace("μ", "u")
    s = "".join(ch for ch in s if ch.isprintable())
    s = _strip_accents(s)            # <— quitamos tildes
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = _canon(c)
        # 1) coincidencia directa
        if k in _def_map:
            ren[c] = _def_map[k]; continue
        # 2) reglas heurísticas
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
            ren[c] = "Clorofila (μg/L)"; continue
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
    """Convierte strings con separadores locales (miles/decimales) a float de forma robusta."""
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

DEFAULT_EPS = (0.3, 1.0, 5.0)

def fuzzy_memberships_scalar(x, eps=DEFAULT_EPS):
    e1, e2, e3 = eps
    m0 = _trapezoid(x, 0.0, 0.0, 2.0 - e1, 2.0 + e1)
    m1 = _trapezoid(x, 2.0 - e1, 2.0 + e1, 7.0 - e2, 7.0 + e2)
    m2 = _trapezoid(x, 7.0 - e2, 7.0 + e2, 40.0 - e3, 40.0 + e3)
    m3 = _right_shoulder(x, 40.0 - e3, 40.0 + e3)
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
        if lab in idx_map: out[:, j] = proba[:, idx_map[lab]]
        else: out[:, j] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, out=out, where=row_sums > 0)
    return out

# ------------------------- 1) Tabla AMSA -------------------------
DEFAULT_DIR_AMSA = Path("datasets_lagos")
DEFAULT_DIR_POND = Path("pruebas_piloto")
DEFAULT_AMSA = DEFAULT_DIR_AMSA / "DATOS AMSA.csv"

st.subheader("📄 Datos AMSA — vista inicial")

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

# Normaliza encabezados
df_amsa = normalize_columns(df_amsa)

# Renombrado **silencioso** de Oxígeno Disuelto si falta el nombre canónico
if "Oxígeno Disuelto (mg/L)" not in df_amsa.columns:
    # Buscar cualquier columna que parezca oxígeno disuelto (robusto a tildes/mayúsculas)
    cands = []
    for col in df_amsa.columns:
        kc = _canon(col)
        if (("oxigeno" in kc or "oxygen" in kc or "dissolved oxygen" in kc
             or re.search(r"\bdo\b", kc) or re.search(r"\bod\b", kc) or "o2" in kc)
            and ("mg/l" in kc or "mg l" in kc or "mg" in kc)):
            cands.append(col)
    if cands:
        df_amsa.rename(columns={cands[0]: "Oxígeno Disuelto (mg/L)"}, inplace=True)
    # Si no hay candidatas, no molestamos al usuario: se verá el error solo en la validación final.

# Vista previa (10 filas) + expander
st.markdown("**Vista previa (10 primeras filas):**")
st.table(df_amsa.head(10))
with st.expander("⬇️ Ver todos los datos AMSA"):
    st.dataframe(df_amsa, use_container_width=True)

# Conversión numérica robusta
for c in REQ_FEATURES + [TARGET]:
    if c in df_amsa.columns:
        df_amsa[c] = to_numeric_smart(df_amsa[c])

# Validación de columnas (si falta algo, mostramos el error normal aquí)
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
        st.warning("TensorFlow/Keras no está disponible. Se mostrará una curva ficticia.")
        losses = np.linspace(1.0, 0.2, 60) + 0.05*np.random.randn(60)
        val_losses = losses + 0.05*np.random.randn(60) + 0.05
        fig_loss, ax = plt.subplots()
        ax.plot(losses, label="Pérdida entrenamiento")
        ax.plot(val_losses, label="Pérdida validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax.set_title("Curva de entrenamiento (simulada)")
        ax.grid(True); ax.legend(); fig_loss.tight_layout()
        st.pyplot(fig_loss, use_container_width=True)
        TRAIN_SCALER = None
        TRAIN_MODEL = None
        TRAIN_Y_LOG1P = False
    else:
        # --- Split
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

        # --- Escalado de X
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # --- Transformación del objetivo
        Y_LOG1P = True
        y_tr_t = np.log1p(y_tr) if Y_LOG1P else y_tr
        y_te_t = np.log1p(y_te) if Y_LOG1P else y_te

        # --- Modelo
        model = keras.Sequential([
            layers.Input(shape=(X_tr_s.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])

        # --- Pérdida robusta + callbacks
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.Huber(delta=1.0)
        )
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True, verbose=0
        )
        rl = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=0
        )

        hist = model.fit(
            X_tr_s, y_tr_t,
            validation_data=(X_te_s, y_te_t),
            epochs=400, batch_size=32, verbose=0,
            callbacks=[es, rl]
        )

        # --- Curva
        fig_loss, ax = plt.subplots()
        ax.plot(hist.history["loss"], label="Pérdida entrenamiento")
        ax.plot(hist.history["val_loss"], label="Pérdida validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.set_title("Curva de entrenamiento (Regresión NN sobre AMSA)")
        ax.grid(True); ax.legend(); fig_loss.tight_layout()
        # ax.set_yscale("log")  # <- opcional
        st.pyplot(fig_loss, use_container_width=True)

        # Guardar para inferencia posterior
        TRAIN_SCALER = scaler
        TRAIN_MODEL = model
        TRAIN_Y_LOG1P = Y_LOG1P

with col_note:
    st.info(
        """
        **Nota breve:** La curva muestra cómo evoluciona la *pérdida* durante el entrenamiento
        y validación del modelo de **regresión** que estima la clorofila (μg/L)
        a partir de pH, temperatura, conductividad, oxígeno disuelto y turbidez (datos AMSA).
        Una curva descendente y estable sugiere buen ajuste sin sobreajuste.
        """
    )
    user_note = st.text_area(
        "✍️ Puedes editar esta explicación:",
        value="La pérdida de validación converge sin aumentar, indicando buen generalizado."
    )


# ------------------------- 3) Matrices difusas (SVM y KNN) + Nota -------------------------
st.subheader("🧩 Matrices de confusión **difusas** con AMSA (SVM y KNN)")

X_train, X_test, y_train_num, y_test_num = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
y_train_cls = pd.cut(y_train_num, bins=BINS, labels=LABELS, right=False)

svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale",
                                             class_weight="balanced", probability=True, random_state=42))
knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))

svm_clf.fit(X_train, y_train_cls)
knn_clf.fit(X_train, y_train_cls)

proba_svm = svm_clf.predict_proba(X_test)
proba_knn = knn_clf.predict_proba(X_test)

svm_classes = svm_clf.named_steps[list(svm_clf.named_steps.keys())[-1]].classes_
knn_classes = knn_clf.named_steps[list(knn_clf.named_steps.keys())[-1]].classes_

proba_svm_al = align_proba_to_labels(proba_svm, svm_classes, LABELS)
proba_knn_al = align_proba_to_labels(proba_knn, knn_classes, LABELS)

cm_svm_fuzzy = fuzzy_confusion_from_probs(y_test_num, proba_svm_al, n_classes=4)
cm_knn_fuzzy = fuzzy_confusion_from_probs(y_test_num, proba_knn_al, n_classes=4)

c1, c2 = st.columns(2)
with c1:
    st.pyplot(plot_confusion_matrix_pretty_float(cm_svm_fuzzy, LABELS, "Matriz **difusa** — SVM (AMSA)"),
              use_container_width=True)
    st.caption(f"Suma de pesos (SVM): {cm_svm_fuzzy.sum():.2f}")
with c2:
    st.pyplot(plot_confusion_matrix_pretty_float(cm_knn_fuzzy, LABELS, "Matriz **difusa** — KNN (AMSA)"),
              use_container_width=True)
    st.caption(f"Suma de pesos (KNN): {cm_knn_fuzzy.sum():.2f}")

st.info(
    """
    **Nota breve:** Estas matrices **difusas** consideran la cercanía a los umbrales (2, 7, 40 μg/L).
    En lugar de contar aciertos/errores duros, reparten *peso* entre clases vecinas cuando la
    clorofila real está cerca de un límite. Así, penalizan menos las confusiones razonables.
    """
)
user_note2 = st.text_area("✍️ Puedes editar esta explicación de las matrices:",
                          value="La matriz difusa suaviza el conteo cerca de 2, 7 y 40 μg/L.")

# ------------------------- 4) Botón: Predecir con datos del estanque -------------------------
st.subheader("🧪 Predicción y matrices (difusas) con datos del estanque")

clicked = st.button("🔮 Predecir con datos del estanque")
if clicked:
    # 1) Ruta / carga del CSV del estanque
    DEFAULT_POND = DEFAULT_DIR_POND / "dataframe1.csv"
    pond_path_input = st.text_input("Ruta a **dataframe del estanque**", value=str(DEFAULT_POND), key="pond_path")
    pond_path = Path(pond_path_input)

    if not pond_path.exists():
        st.warning("No encuentro **dataframe1.csv** en la ruta indicada. Sube el archivo:")
        up2 = st.file_uploader("Sube dataframe1.csv", type=["csv"], key="pond")
        if up2 is None:
            st.stop()
        df_pond = read_csv_robust(up2)
    else:
        df_pond = read_csv_robust(pond_path)

    # 2) Normalización de encabezados y numéricos
    df_pond = normalize_columns(df_pond)
    have_true = TARGET in df_pond.columns

    for c in REQ_FEATURES + ([TARGET] if have_true else []):
        if c in df_pond.columns:
            df_pond[c] = to_numeric_smart(df_pond[c])

    df_pond = df_pond.dropna(subset=REQ_FEATURES).reset_index(drop=True)
    if df_pond.empty:
        st.error("El archivo del estanque no tiene filas válidas tras limpieza.")
        st.stop()

    # 3) Matrices con SVM/KNN (probabilidades sobre X del estanque)
    Xp = df_pond[REQ_FEATURES].values

    proba_svm_p = svm_clf.predict_proba(Xp)
    proba_knn_p = knn_clf.predict_proba(Xp)
    proba_svm_p_al = align_proba_to_labels(proba_svm_p, svm_classes, LABELS)
    proba_knn_p_al = align_proba_to_labels(proba_knn_p, knn_classes, LABELS)

    # 4) "Verdad" para la matriz difusa:
    #    - si el estanque trae clorofila real, úsala
    #    - si no, intenta la red entrenada en la sección 2 (con des-transformación si log1p)
    #    - si tampoco hay red, usa centroides por clase (fallback)
    if have_true:
        y_true_p = pd.to_numeric(df_pond[TARGET], errors="coerce").fillna(0).to_numpy()
        used_proxy = False
    else:
        tm  = globals().get("TRAIN_MODEL", None)
        ts  = globals().get("TRAIN_SCALER", None)
        ylg = globals().get("TRAIN_Y_LOG1P", False)

        if KERAS_OK and (tm is not None) and (ts is not None):
            Xp_s = ts.transform(Xp)
            y_pred_t = tm.predict(Xp_s, verbose=0).ravel()
            y_proxy  = np.expm1(y_pred_t) if ylg else y_pred_t
            y_true_p = np.clip(y_proxy, 0.0, None)
            used_proxy = True
        else:
            pred_cls = np.argmax(proba_svm_p_al, axis=1)
            centers  = np.array([1.0, 4.5, 20.0, 60.0])  # centroides aproximados de rangos
            y_true_p = centers[pred_cls]
            used_proxy = True

    # 5) Matrices difusas con esas "verdades"
    cm_svm_p = fuzzy_confusion_from_probs(y_true_p, proba_svm_p_al, n_classes=4)
    cm_knn_p = fuzzy_confusion_from_probs(y_true_p, proba_knn_p_al, n_classes=4)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.pyplot(
            plot_confusion_matrix_pretty_float(cm_svm_p, LABELS, "Matriz **difusa** — SVM (Estanque)"),
            use_container_width=True
        )
        st.caption(f"Suma de pesos (SVM): {cm_svm_p.sum():.2f}")
    with cc2:
        st.pyplot(
            plot_confusion_matrix_pretty_float(cm_knn_p, LABELS, "Matriz **difusa** — KNN (Estanque)"),
            use_container_width=True
        )
        st.caption(f"Suma de pesos (KNN): {cm_knn_p.sum():.2f}")

    if used_proxy and not have_true:
        st.caption("ℹ️ Se usó **proxy** de clorofila para la matriz difusa (no había columna de clorofila real).")

    st.success("Listo. Matrices del estanque generadas.")
    # ========= Predicciones continuas para exportar =========
    # Intentamos usar la NN de regresión entrenada (sección 2).
    # Si no existe, usamos una estimación por SVM (valor esperado con centroides).
    tm  = globals().get("TRAIN_MODEL", None)
    ts  = globals().get("TRAIN_SCALER", None)
    ylg = globals().get("TRAIN_Y_LOG1P", False)

    if KERAS_OK and (tm is not None) and (ts is not None):
        Xp_s = ts.transform(Xp)
        yhat_t = tm.predict(Xp_s, verbose=0).ravel()
        yhat   = np.expm1(yhat_t) if ylg else yhat_t
    else:
        # Esperanza de clorofila a partir de probabilidades SVM y centroides de cada rango
        centers = np.array([1.0, 4.5, 20.0, 60.0])
        yhat = proba_svm_p_al @ centers

    yhat = np.clip(yhat, 0.0, None)

    # DataFrame a exportar
    df_pred_export = df_pond.copy()
    df_pred_export["Clorofila_predicha (μg/L)"] = yhat

    # ========= Botones inferiores: Volver (izq) / Descargar CSV (der) =========
    bot_left, bot_right = st.columns(2)
    with bot_left:
        # OPCIÓN A (multi-página nativa de Streamlit 1.27+):
        volver = st.button("⬅️ Volver", use_container_width=True)
        if volver:
            # Cambia el path a tu página principal (Home) según tu estructura:
            #   - si está en /pages/Menu.py:  st.switch_page("pages/Menu.py")
            #   - si es el script principal:  st.switch_page("Home.py")
            st.switch_page("Home.py")

        # OPCIÓN B (si NO usas multi-página nativa):
        # Usa un flag de session_state para volver a tu “menú” propio.
        # if st.button("⬅️ Volver", use_container_width=True):
        #     st.session_state["page"] = "menu"  # tu lógica de navegación
        #     st.experimental_rerun()

    with bot_right:
        st.download_button(
            "⬇️ Descargar predicciones (.csv)",
            data=df_pred_export.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_estanque.csv",
            mime="text/csv",
            use_container_width=True,
        )

