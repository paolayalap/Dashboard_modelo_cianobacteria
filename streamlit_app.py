import streamlit as st
import pandas as pd

st.title('🧪 Dashboard cyanobacteria')

st.info('This is an app to demostrate and analize the data to create a system to detect cyanobacteria in water bodies with machine learning.')
st.write('First, we're going to present our reference data. This is going to have parameters like: pH, temperature, dissolve oxygen, conductivity and turbidity. ')

with st.expander('Data IBAGUA'):
  st.write('**In 2017 and 2019, Evelyn Rodas take some measures of clorofilla type a to know how many fitoplancton was in the lake of Amatitlán.
  She shared to us this information to create our first models**')
  df = pd.read_csv('https://raw.githubusercontent.com/paolayalap/Dashboard_modelo_cianobacteria/refs/heads/master/datos_lago.csv')
  df
