import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model
model_path = "C:/users/UTS_datming/model_diabetes1.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load dataset untuk referensi fitur
dataset_path = "C:/users/UTS_datming/dataset_fix1.csv"
df = pd.read_csv(dataset_path)

# Ambil nama fitur dari dataset
features = df.columns[:-1]  # Asumsi kolom terakhir adalah label

# Judul aplikasi
st.markdown("""
    <h1 style='text-align: center;'> 🩺GlucoPredict</h1>
    <p style='text-align: justify;'>Model ini menggunakan <a href='https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database'>Pima Indians Diabetes Dataset</a> untuk membantu tenaga medis dalam menganalisis risiko diabetes pasien.</p>
""", unsafe_allow_html=True)
st.image("gambarcekguladarah.png")  # Menambahkan gambar di bawah judul

st.markdown("---")  # Garis pemisah

# Layout dengan 4 kolom
cols = st.columns(4)

# Buat input form
input_data = []
for i, feature in enumerate(features):
    feature_clean = feature.strip().lower()
    with cols[i % 4]:  # Distribusi input ke 4 kolom
        if feature_clean == "gender":
            value = st.selectbox("Masukkan Gender", ["Female", "Male"], key=feature)
            value = 1 if value == "Male" else 0  # 1 untuk Male, 0 untuk Female
        elif feature_clean in ["hypertension", "heart_disease"]:
            value = st.selectbox(f"Masukkan {feature}", ["No", "Yes"], key=feature)
            value = 1 if value == "Yes" else 0  # 1 untuk Yes, 0 untuk No
        elif feature_clean == "smoking_history":
            value = st.selectbox("Masukkan smoking history", ["Never", "Current", "Ever", "Former", "Not Current"], key=feature)
            smoking_values = {
                "smoking_history_current": 1 if value == "Current" else 0,
                "smoking_history_ever": 1 if value == "Ever" else 0,
                "smoking_history_former": 1 if value == "Former" else 0,
                "smoking_history_never": 1 if value == "Never" else 0,
                "smoking_history_not current": 1 if value == "Not Current" else 0
            }
            input_data.extend(smoking_values.values())
            continue  # Lewati appending langsung karena sudah ditambahkan ke list
        else:
            value = st.number_input(f"Input {feature}", value=0.0, step=0.1, key=feature)
        input_data.append(value)

# Tombol Prediksi
st.markdown("---")  # Garis pemisah
if st.button("Prediksi"):
    # Konversi input menjadi array
    input_array = np.array(input_data).reshape(1, -1)
    
    # Lakukan prediksi
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[0][1] * 100  # Probabilitas terkena diabetes
    
    # Tampilkan hasil
    if prediction[0] == 1:
        st.error(f"Hasil: Orang ini kemungkinan besar terkena diabetes.")
    else:
        st.success(f"Hasil: Orang ini kemungkinan besar tidak terkena diabetes.")
        
# Tambahkan copyright
st.markdown("© Kelompok 6 UTS Data Mining")