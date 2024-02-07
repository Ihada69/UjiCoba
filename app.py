import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk memuat model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

st.set_page_config(page_title="Prediksi Harga Emas", page_icon=":money_with_wings:")

st.title("Prediksi Harga Emas")
st.write("Prediksi harga emas berdasarkan IHSG, Inflasi, dan Kurs Dollar menggunakan model Decision Tree, Random Forest, atau AdaBoost.")

# Input pengguna di sidebar
st.sidebar.header('Input Parameter')
ihsg_close = st.sidebar.number_input('IHSG Close', format="%.2f")
kurs_jual = st.sidebar.number_input('Kurs Jual', format="%.2f")
kurs_beli = st.sidebar.number_input('Kurs Beli', format="%.2f")
data_inflasi = st.sidebar.number_input('Data Inflasi (dalam persen, contoh: masukkan 3.5 untuk 3,5%)', format="%.2f")
inflasi_desimal = data_inflasi / 100  # Konversi inflasi ke bentuk desimal

# Membuat DataFrame dari input pengguna
input_df = pd.DataFrame([[ihsg_close, kurs_jual, kurs_beli, inflasi_desimal]], columns=['IHSG Close', 'Kurs Jual', 'Kurs Beli', 'Data Inflasi'])

model_option = st.selectbox("Pilih Model untuk Prediksi:", ['Decision Tree', 'Random Forest', 'AdaBoost'])

# Memuat model berdasarkan pilihan pengguna
model_paths = {'Decision Tree': 'decision_tree_tuned.pkl', 'Random Forest': 'random_forest_tuned.pkl', 'AdaBoost': 'adaboost_tuned.pkl'}
model = load_model(model_paths[model_option])

# Tombol prediksi
if st.button("Prediksi Harga"):
    predicted_price = model.predict(input_df)[0]
    st.write("")
    st.subheader(f"Harga Emas yang Diprediksi: Rp {predicted_price:,.2f} menggunakan model {model_option}")
