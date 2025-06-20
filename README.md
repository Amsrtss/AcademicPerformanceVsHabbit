# 🎓 Student Habits vs Academic Performance Prediction

Aplikasi interaktif berbasis Streamlit untuk memprediksi skor ujian mahasiswa berdasarkan kebiasaan harian seperti jam belajar, kehadiran, tidur, penggunaan media sosial, dan lainnya. Proyek ini menggunakan Linear Regression dan dataset dari [Kaggle](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance).

---

## 📊 Fitur Utama

- Menampilkan dataset asli dan hasil preprocessing
- Visualisasi evaluasi model regresi
- Form interaktif untuk prediksi skor akhir mahasiswa
- Model dapat dipakai langsung oleh pengguna dengan input kebiasaan harian

---

## 📁 Struktur Proyek

```

├── app.py                      # Aplikasi Streamlit
├── train\_model.py             # Script pelatihan dan evaluasi model
├── preprocessing.ipynb        # Notebook preprocessing data
├── data.csv                   # Dataset mentah dari Kaggle
├── df\_encoded.csv             # Dataset setelah encoding
├── linear\_regression\_model.pkl # Model terlatih
├── scaler.pkl                 # Scaler data
├── evaluation\_results.pkl     # Hasil evaluasi model
├── requirements.txt           # Daftar pustaka Python
└── README.md                  # Dokumentasi proyek ini

````

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone repositori:**

```bash
git clone https://github.com/Amsrtss/AcademicPerformanceVsHabit.git
cd AcademicPerformanceVsHabit
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Jalankan aplikasi Streamlit:**

```bash
streamlit run app.py
```

---

## 🔬 Proses Model

* **Preprocessing:**

  * Deteksi dan penanganan missing value
  * Encoding ordinal dan nominal
  * Standarisasi fitur numerik
* **Modeling:**

  * Linear Regression
  * Split data: 72% train, 8% validation, 20% test
  * Evaluasi menggunakan MSE dan R²
* **Deployment:**

  * Aplikasi Streamlit untuk demo prediksi interaktif

---

## 📈 Hasil Evaluasi Model

| Dataset    | MSE              | R² |
| ---------- | ---------------- | -- |
| Validation | Tertampil di app |    |
| Test       | Tertampil di app |    |

*Hasil evaluasi akan muncul otomatis saat aplikasi dijalankan.*

---

## 📄 Lisensi

MIT License. Silakan gunakan dan modifikasi proyek ini sesuai kebutuhan.


