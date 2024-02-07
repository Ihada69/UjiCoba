import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk memuat model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Membuat sidebar untuk input parameter
st.sidebar.header('Input Parameter')

def user_input_features():
    ihsg_close = st.sidebar.number_input('IHSG Close')
    kurs_jual = st.sidebar.number_input('Kurs Jual')
    kurs_beli = st.sidebar.number_input('Kurs Beli')
    data_inflasi = st.sidebar.number_input('Data Inflasi', format="%f")
    data = {'IHSG Close': ihsg_close,
            'Kurs Jual': kurs_jual,
            'Kurs Beli': kurs_beli,
            'Data Inflasi': data_inflasi}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parameter Input Pengguna')
st.write(df)

# Memuat model (ganti 'model_path' dengan path yang sesuai)
model_rf = load_model('random_forest_tuned.pkl')
model_dt = load_model('decision_tree_tuned.pkl')
model_ab = load_model('adaboost_tuned.pkl')

# Membuat prediksi
prediction_rf = model_rf.predict(df)
prediction_dt = model_dt.predict(df)
prediction_ab = model_ab.predict(df)

st.subheader('Prediksi Harga Emas')
st.write(f'Random Forest: {prediction_rf[0]}')
st.write(f'Decision Tree: {prediction_dt[0]}')
st.write(f'AdaBoost: {prediction_ab[0]}')
