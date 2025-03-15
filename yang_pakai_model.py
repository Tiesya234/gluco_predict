import streamlit as st
import numpy as np
import joblib
import os

# Cek apakah model tersedia
model_path = "C:\\Users\\UTS_datming\\model_diabetes.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model tidak ditemukan! Pastikan file `model_diabetes.pkl` tersedia.")
    st.stop()

# Load model yang sudah dilatih
model = joblib.load(model_path)

# Judul aplikasi
st.title("🩺 GlucoPredict")
st.write("Model ini menggunakan **[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** untuk membantu tenaga medis dalam menganalisis risiko diabetes pasien.")


# Layout input
col1, col2 = st.columns(2)

# Input Gender (Male/Female)
gender = col1.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0  # Male = 1, Female = 0

# Input Age
age = col2.number_input("Age (tahun)", min_value=0, max_value=120, value=45)

# Input Hypertension (Yes/No)
hypertension = col1.selectbox("Hypertension", ["No", "Yes"])
hypertension = 1 if hypertension == "Yes" else 0

# Input Heart Disease (Yes/No)
heart_disease = col2.selectbox("Heart Disease", ["No", "Yes"])
heart_disease = 1 if heart_disease == "Yes" else 0

# Input BMI
bmi = col1.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=22.5)

# Input HbA1c Level
hba1c = col2.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)

# Input Blood Glucose Level
blood_glucose = col1.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=300, value=110)

# Input Smoking History (One-Hot Encoding)
smoking_options = ["Current", "Ever", "Former", "Never", "Not current"]
smoking_history = col2.selectbox("Smoking History", smoking_options)

# Konversi Smoking History menjadi One-Hot Encoding
smoking_encoded = [1 if smoking_history == option else 0 for option in smoking_options]

# Prediksi
st.markdown("---")  # Garis pemisah
if st.button("🔍 Prediksi Diabetes"):
    # Gabungkan semua input menjadi satu array
    input_data = np.array([gender, age, hypertension, heart_disease, bmi, hba1c, blood_glucose] + smoking_encoded).reshape(1, -1)

    # Prediksi menggunakan model
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probabilitas diabetes

    # Output hasil prediksi
    if prediction[0] == 1:  
        st.success(f"❌ Hasil Prediksi: **Pasien berisiko Diabetes** (Probabilitas: {probability:.2%})")
    else:  
        st.error(f"✅ Hasil Prediksi: **Pasien tidak berisiko Diabetes** (Probabilitas: {probability:.2%})")


    # Rekomendasi medis
    if probability > 0.7:
        st.warning("⚠️ **Rekomendasi:** Konsultasikan dengan dokter dan lakukan tes lebih lanjut seperti **HbA1c atau OGTT**.")
    elif probability > 0.4:
        st.info("🔎 **Rekomendasi:** Perhatikan pola makan dan aktivitas fisik untuk pencegahan.")
