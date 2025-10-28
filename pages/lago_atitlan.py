# ============================================
# Streamlit: CEA — Clorofila (solo)
# - Entrena con DATOS CEA.csv
# - Predice Clorofila en el estanque (NN continua)
# - Matrices de confusión DIFUSAS (CEA y Estanque)
#    * Clorofila: 4 clases (0–2, 2–7, 7–40, ≥40)
#    * En ESTANQUE: "verdad" = predicción continua de NN (TODAS las filas)
#      => suma de pesos = N_filas_estanque
# ============================================

import os, re, unicodedata
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

# ====== NN (para regresión continua) ======
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    KERAS_OK = True
except Exception:
    KERAS_OK = False

# ------------------------- Config UI -------------------------
st.set_page_config(page_title="CEA — Clorofila (difusa)", layout="wide")
st.title("🧪 CEA — Entrenamiento y Predicción (Clorofila)")
st.caption("Entrena con **DATOS CEA.csv** → Predice en el **estanque** con **NN** → Calcula **matrices difusas** (SVM/KNN).")

# ------------------------- Utilidades -------------------------
REQ_FEATURES = ["pH","Temperatura (°C)","Conductividad (μS/cm)","Oxígeno Disuelto (mg/L)","Turbidez (NTU)"]
TARGET_CHL = "Clorofila (μg/L)"

BINS_CHL   = [0, 2, 7, 40, np.inf]
LABELS_CHL = ["Muy bajo (0–2)","Bajo (2–7)","Moderado (7–40)","Muy alto (≥40)"]

# Mapeo flexible de nombres
_def_map = {
    "ph":"pH",
    "temperatura":"Temperatura (°C)","temperatura (c)":"Temperatura (°C)","temp (°c)":"Temperatura (°C)","temp":"Temperatura (°C)",
    "conductividad (us/cm)":"Conductividad (μS/cm)","conductividad(us/cm)":"Conductividad (μS/cm)","conductividad (s/cm)":"Conductividad (μS/cm)","conductividad":"Conductividad (μS/cm)",
    "oxígeno disuelto (mg/l)":"Oxígeno Disuelto (mg/L)","oxigeno disuelto (mg/l)":"Oxígeno Disuelto (mg/L)","oxígeno disuelto (mgl)":"Oxígeno Disuelto (mg/L)","oxigeno disuelto (mgl)":"Oxígeno Disuelto (mg/L)",
    "oxygen dissolved (mg/l)":"Oxígeno Disuelto (mg/L)","dissolved oxygen (mg/l)":"Oxígeno Disuelto (mg/L)","do (mg/l)":"Oxígeno Disuelto (mg/L)","od (mg/l)":"Oxígeno Disuelto (mg/L)","o2 disuelto (mg/l)":"Oxígeno Disuelto (mg/L)",
    "clorofila (μg/l)":TARGET_CHL,"clorofila (ug/l)":TARGET_CHL,"clorofila":TARGET_CHL,"chlorophyll a":TARGET_CHL,"chlorophyll-a":TARGET_CHL,
}

def _strip_accents(text: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', text) if not unicodedata.combining(ch))

def _canon(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("µ", "u").replace("μ", "u")
    s = "".join(ch for ch in s if ch.isprintable())
    s = _strip_accents(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = _canon(c)
        if k in _def_map: ren[c] = _def_map[k]; continue
        if "conductividad" in k and ("us/cm" in k or "uscm" in k or "s/cm" in k): ren[c] = "Conductividad (μS/cm)"; continue
        if "temperatura" in k or k.startswith("temp"): ren[c] = "Temperatura (°C)"; continue
        if "turbidez" in k or "turbiedad" in k or "ntu" in k: ren[c] = "Turbidez (NTU)"; continue
