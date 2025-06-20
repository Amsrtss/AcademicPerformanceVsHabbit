import streamlit as st
import joblib
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

#Load data dan model
df = pd.read_csv("data.csv")
df_clean = pd.read_csv("df_encoded.csv")
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
# Load hasil evaluasi
results = joblib.load('evaluation_results.pkl')

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Data", "Evaluasi Model", "Prediksi"])

st.title("Student Habits vs Academic Performance")
st.write("Ever wondered how much Netflix, sleep, or TikTok scrolling affects your grades? 👀 This dataset simulates 1,000 students' daily habits—from study time to mental health—and compares them to final exam scores. It's like spying on your GPA through the lens of lifestyle.")

st.header("Data yang digunakan")
st.write("Menampilkan data kebiasaan mahasiswa sebelum dan sesudah proses pembersihan serta encoding, sebagai dasar pembuatan model prediksi akademik. Dataset yang digunakan bersumber dari Kaggle dengan link berikut: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance?resource=download")
st.subheader("Dataset")
st.write("Dataset ini memiliki 15 fitur dan 1000 baris data.")
st.dataframe(df)
st.write("Setelah preprocessing:")
st.dataframe(df_clean)
st.write("Dataset menjadi berubah dengan 15 fitur dan 909 baris data. Kemudian dilakukan pemisahan data untuk training (72%), validasi (8%), dan testing (20%). Kemudian dilakukan modelling dengan algoritma Regresi Linear.")

st.subheader("Hasil evaluasi model")
st.write("Visualisasi performa model regresi berdasarkan metrik MSE dan R² pada data validasi dan pengujian, bertujuan untuk mengetahui performa dari model yang sudah dilatih. Apakah model sudah layak atau butuh training ulang.")
# Ambil data evaluasi
val_mse = results["validation"]["mse"]
val_r2 = results["validation"]["r2"]
test_mse = results["test"]["mse"]
test_r2 = results["test"]["r2"]

# Buat DataFrame evaluasi untuk visualisasi
eval_df = pd.DataFrame({
    "MSE": [val_mse, test_mse],
    "R²": [val_r2, test_r2]
}, index=["Validation", "Test"])

# Bar chart dengan matplotlib
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# MSE plot
ax[0].bar(eval_df.index, eval_df["MSE"], color=["skyblue", "salmon"])
ax[0].set_title("Mean Squared Error")
ax[0].set_ylabel("MSE")
ax[0].set_ylim(0, max(eval_df["MSE"]) * 1.2)

# R² plot
ax[1].bar(eval_df.index, eval_df["R²"], color=["lightgreen", "orange"])
ax[1].set_title("R² Score")
ax[1].set_ylabel("R²")
ax[1].set_ylim(0, 1)

# Tampilkan plot di Streamlit
st.pyplot(fig)

# Tambahkan deskripsi hasil
st.write("**Interpretasi Singkat:**")
st.markdown(f"""
- Validation MSE: `{val_mse:.2f}`, R²: `{val_r2:.2f}`  
- Test MSE: `{test_mse:.2f}`, R²: `{test_r2:.2f}`  

Semakin kecil nilai MSE dan semakin mendekati 1 nilai R², maka semakin baik performa model regresi.
""")

st.header("Coba Prediksi Skor Akhir Mahasiswa")
st.write("Formulir yang dapat diisi untuk memasukkan data kebiasaan mahasiswa, seperti jam belajar, kehadiran, dan kualitas tidur, guna memprediksi skor ujian menggunakan model yang telah dilatih.")

with st.form("prediction_form"):
    age = st.number_input("Usia", min_value=15, max_value=40, value=21)
    study_hours = st.slider("Jam belajar per hari", 0.0, 10.0, 2.0)
    social_media = st.slider("Jam sosial media per hari", 0.0, 10.0, 2.0)
    netflix = st.slider("Jam nonton Netflix per hari", 0.0, 10.0, 2.0)
    attendance = st.slider("Persentase kehadiran (%)", 0.0, 100.0, 90.0)
    sleep = st.slider("Jam tidur per hari", 0.0, 12.0, 7.0)
    diet = st.selectbox("Kualitas diet (0: buruk, 1: sedang, 2: baik)", options=[0, 1, 2])  
    exercise = st.selectbox("Frekuensi olahraga (misal: 0–7 hari/minggu)", options=range(0, 8))  
    parental_edu = st.selectbox("Tingkat pendidikan orang tua (0: SMA/sederajat, 1: sarjana, 2: magister)", options=[0, 1, 2])  
    internet = st.selectbox("Kualitas internet (0: buruk, 1: sedang, 2: baik)", options=[0, 1, 2])  
    mental = st.slider("Rating kesehatan mental", 0.0, 10.0, 5.0)

    gender = st.radio("Jenis kelamin", options=["Male", "Female", "Other"])
    part_time = st.checkbox("Punya pekerjaan paruh waktu")
    extracurricular = st.checkbox("Ikut kegiatan ekstrakurikuler")

    submitted = st.form_submit_button("Prediksi Skor")

# Proses prediksi
if submitted:
    # Susun input sesuai urutan fitur model
    input_dict = {
        'age': age,
        'study_hours_per_day': study_hours,
        'social_media_hours': social_media,
        'netflix_hours': netflix,
        'attendance_percentage': attendance,
        'sleep_hours': sleep,
        'diet_quality': diet,
        'exercise_frequency': exercise,
        'parental_education_level': parental_edu,
        'internet_quality': internet,
        'mental_health_rating': mental,
        'gender_Male': gender == "Male",
        'gender_Other': gender == "Other",
        'part_time_job_Yes': part_time,
        'extracurricular_participation_Yes': extracurricular
    }

    input_df = pd.DataFrame([input_dict])

    # Konversi boolean ke float (agar sesuai data latih)
    input_df = input_df.astype(float)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    # Batasi skor ke rentang 0–100
    prediction = max(0, min(100, prediction))

    # Tampilkan hasil
    st.success(f"Prediksi Skor Ujian: **{prediction:.2f}**")


