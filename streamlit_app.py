
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Evita ciertos warnings numéricos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import streamlit as st
import pathlib, re
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib.patches import Rectangle


st.title("🧪 Dashboard cyanobacteria")
st.caption("Resultados del modelo")


