import streamlit as st
import pandas as pd

st.title('🙌 Machine Learning App')

st.write('This is an app builds a machine learning model')

df = pd.read_csv('https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/refs/heads/master/datos_lago.csv')
df
