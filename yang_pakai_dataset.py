import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("C:/users/UTS_datming/dataset_fix.csv")  # Pastikan path benar

# Encode data kategorikal
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:  
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Pisahkan fitur dan target
X = df.iloc[:, :-1]  # Semua kolom kecuali target sebagai fitur
y = df.iloc[:, -1]   # Kolom target

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi dan latih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Simpan model dan scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Streamlit Dashboard
st.title("🩺 GlucoPredict")
st.write("Model ini menggunakan **[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** untuk membantu tenaga medis dalam menganalisis risiko diabetes pasien.")

# Dictionary satuan dan contoh input
satuan_dan_contoh = {
    "Pregnancies": ("jumlah", 2),
    "Glucose": ("mg/dL", 110),
    "BloodPressure": ("mmHg", 80),
    "SkinThickness": ("mm", 20),
    "Insulin": ("μU/mL", 85),
    "BMI": ("kg/m²", 22.5),
    "DiabetesPedigreeFunction": ("(tanpa satuan, 0-2)", 0.5),
    "Age": ("tahun", 45)
}

# Membagi input ke dalam 3 kolom agar lebih rapi
col1, col2, col3 = st.columns(3)

# Input user dengan nilai default dan bantuan dalam layout horizontal
input_data = []
for i, col_name in enumerate(X.columns):
    satuan, contoh_default = satuan_dan_contoh.get(col_name, ("", 0.0))

    # Tentukan di kolom mana input akan ditampilkan
    if i % 3 == 0:
        col = col1
    elif i % 3 == 1:
        col = col2
    else:
        col = col3

    # Jika data kategorikal, pakai dropdown
    if col_name in label_encoders:
        options = list(label_encoders[col_name].classes_)
        selected_option = col.selectbox(f"{col_name}", options)
        encoded_value = label_encoders[col_name].transform([selected_option])[0]
        input_data.append(encoded_value)
    else:
        val = col.number_input(
            f"{col_name} ({satuan})", 
            value=contoh_default,  
            help=f"Contoh: {contoh_default}"  
        )
        input_data.append(val)

# Prediksi
st.markdown("---")  # Garis pemisah
if st.button("🔍 Prediksi"):
    if os.path.exists("diabetes_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")

        input_array = scaler.transform([input_data])
        prediction = model.predict(input_array)

        st.success("✅ Hasil Prediksi: **Diabetes**" if prediction[0] == 1 else "❌ Hasil Prediksi: **Tidak Diabetes**")
    else:
        st.error("Model belum tersedia. Silakan latih ulang model terlebih dahulu.")
