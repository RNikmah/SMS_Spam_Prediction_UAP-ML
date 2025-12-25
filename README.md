# 📩 Spam Detection System with Deep Learning & Transformer Models

## � Daftar Isi

- [Latar Belakang](#-latar-belakang)
- [Deskripsi Proyek](#-deskripsi-proyek)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Model yang Digunakan](#-model-yang-digunakan)
- [Hasil Evaluasi & Analisis Perbandingan](#-hasil-evaluasi--analisis-perbandingan)
- [Dashboard Streamlit](#-dashboard-streamlit-mini-produk)
- [Panduan Menjalankan Sistem](#-panduan-menjalankan-sistem)
- [Troubleshooting](#-troubleshooting)
- [Fitur Aplikasi](#-fitur-aplikasi)
- [Dependencies](#-dependencies)
- [Catatan Tambahan](#-catatan-tambahan)
- [Author](#-author)
- [License](#-license)
- [Links](#-links)

---

## 🌟 Latar Belakang

Dalam era digital saat ini, komunikasi melalui pesan teks (SMS) dan email telah menjadi bagian integral dari kehidupan sehari-hari. Namun, seiring dengan meningkatnya penggunaan platform komunikasi digital, ancaman dari **spam messages** juga semakin meningkat. Spam tidak hanya mengganggu produktivitas pengguna, tetapi juga dapat menjadi vektor untuk **phishing**, **malware**, dan berbagai bentuk **serangan siber** lainnya.

### Permasalahan

1. **Volume Spam yang Tinggi**: Jutaan pesan spam dikirim setiap hari, memenuhi inbox pengguna dan menyembunyikan pesan penting
2. **Teknik Spam yang Berkembang**: Spammer terus mengembangkan teknik baru untuk melewati filter tradisional
3. **Kerugian Finansial**: Spam dapat menyebabkan kerugian finansial melalui penipuan dan pencurian identitas
4. **Penurunan Produktivitas**: Waktu yang terbuang untuk memilah spam dari pesan legitimate

### Solusi dengan Machine Learning

Pendekatan **Natural Language Processing (NLP)** dan **Deep Learning** menawarkan solusi yang lebih efektif dibandingkan rule-based filtering tradisional. Model pembelajaran mesin dapat:

- **Belajar dari pola kompleks** dalam teks spam
- **Beradaptasi** dengan teknik spam baru
- **Mengklasifikasikan** pesan dengan tingkat akurasi tinggi
- **Mengurangi false positive** (pesan legitimate yang salah diklasifikasi sebagai spam)

### Motivasi Proyek

Proyek ini dimotivasi oleh kebutuhan untuk:

1. **Membandingkan** performa model deep learning tradisional (BiLSTM) dengan transformer-based models (BERT, DistilBERT)
2. **Mengimplementasikan** sistem deteksi spam yang praktis dan dapat digunakan secara real-time
3. **Memahami** trade-off antara kompleksitas model, akurasi, dan efisiensi komputasi
4. **Menyediakan** mini produk yang user-friendly untuk demonstrasi teknologi NLP

---

## �📌 Deskripsi Proyek

Proyek ini bertujuan untuk membangun dan membandingkan sistem klasifikasi pesan **Spam vs Ham** menggunakan tiga pendekatan model pembelajaran mesin, yaitu:

- **Neural Network berbasis BiLSTM** (non-pretrained)
- **BERT** (Pretrained Transformer)
- **DistilBERT** (Lightweight Pretrained Transformer)

Selain evaluasi model, proyek ini juga mengimplementasikan **dashboard berbasis Streamlit** yang memungkinkan pengguna melakukan prediksi secara interaktif dengan visualisasi confidence, threshold, dan perbandingan performa antar model.

**Catatan:** Training model dilakukan di **Google Colab** dengan GPU support, sedangkan deployment aplikasi Streamlit dapat dijalankan secara lokal.

---

## 📂 Dataset & Preprocessing

### Dataset

Dataset yang digunakan adalah **SMS Spam Collection Dataset** dari UCI Machine Learning Repository yang diakses melalui Hugging Face:

🔗 [SMS Spam Collection Dataset](https://huggingface.co/datasets/ucirvine/sms_spam)

Dataset ini merupakan kumpulan pesan SMS yang diklasifikasikan menjadi dua label:

- **Ham** → pesan normal (legitimate messages)
- **Spam** → pesan promosi/penipuan (spam messages)

Dataset dibagi menjadi:
- Training set
- Validation set
- Test set

### Preprocessing

Tahapan preprocessing meliputi:

- Case folding (lowercase)
- Pembersihan karakter khusus
- Tokenisasi teks
- Padding & truncation
- Label encoding

**Catatan:**
- BiLSTM menggunakan tokenizer konvensional
- BERT & DistilBERT menggunakan tokenizer bawaan dari Hugging Face

---

## 🧠 Model yang Digunakan

### 1️⃣ BiLSTM (Non-Pretrained)

- **Arsitektur**: Embedding → BiLSTM → Dense
- **Kelebihan**: ringan dan cepat dilatih
- **Kekurangan**: keterbatasan pemahaman konteks global

### 2️⃣ BERT

- Model pretrained `bert-base-uncased`
- Mampu menangkap konteks dua arah
- Akurasi tinggi, namun komputasi relatif berat

### 3️⃣ DistilBERT

- Versi ringan dari BERT
- Lebih cepat dan efisien
- Akurasi sedikit di bawah BERT namun lebih optimal untuk deployment

---

## 📊 Hasil Evaluasi & Analisis Perbandingan

### 🔍 Metrik Evaluasi

Setiap model dievaluasi menggunakan:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **Grafik Loss & Accuracy**

### 📈 Tabel Perbandingan Performa Model

| Nama Model   | Akurasi | Precision | Recall | F1-Score | Hasil Analisis |
|--------------|---------|-----------|--------|----------|----------------|
| **BiLSTM**   | **0.99** | **0.99** | **0.96** | **0.98** | Model baseline memberikan performa terbaik dengan akurasi dan precision tertinggi, menunjukkan kemampuan yang sangat baik dalam mengklasifikasikan spam meskipun tanpa pretrained weights |
| **BERT**     | 0.98    | 0.98      | 0.92   | 0.95     | Performa sangat baik namun sedikit di bawah BiLSTM, dengan recall terendah yang mengindikasikan beberapa spam messages tidak terdeteksi |
| **DistilBERT** | 0.98  | 0.98      | 0.92   | 0.95     | Performa identik dengan BERT, membuktikan efisiensi knowledge distillation dalam mempertahankan akurasi dengan model yang lebih ringan |

**Kesimpulan:**  
Meskipun BERT dan DistilBERT adalah model pretrained yang lebih kompleks, **BiLSTM** justru memberikan performa terbaik dengan akurasi 99% dan F1-Score 0.98. Hal ini menunjukkan bahwa untuk task klasifikasi spam yang relatif sederhana, arsitektur yang lebih simple namun dioptimalkan dengan baik dapat mengalahkan model transformer yang lebih kompleks. **DistilBERT** tetap menjadi pilihan yang baik untuk production deployment karena trade-off antara performa (sama dengan BERT) dan efisiensi komputasi.

---

## 🖥️ Dashboard Streamlit (Mini Produk)

Fitur utama dashboard:

- ✅ **Dropdown pemilihan model** (BiLSTM / BERT / DistilBERT)
- ✅ **Threshold slider** untuk menentukan batas spam
- ✅ **Visualisasi confidence** (progress bar)
- ✅ **Layout 2 kolom** dengan gradient badges
- ✅ **Riwayat prediksi** (history table dengan expander)
- ✅ **Navigasi multi-page** (Deteksi, Info Model, Evaluasi Model)
- ✅ **Perbandingan performa** antar model dengan visualisasi
- ✅ **Modern UI** dengan gradient header & styling

---

## 🚀 Panduan Menjalankan Sistem

### 📝 Catatan Penting

Proyek ini terbagi menjadi dua bagian:
1. **Training Model** - Dilakukan di Google Colab (sudah selesai, model tersedia)
2. **Deployment Dashboard** - Aplikasi Streamlit untuk inferensi (dapat dijalankan lokal)

### 🎓 Training Model di Google Colab (Opsional)

Jika ingin melakukan training ulang:

1. Buka `Spam_Detection.ipynb` di Google Colab
2. Mount Google Drive untuk menyimpan model
3. Jalankan semua cell secara berurutan
4. Model akan tersimpan di Google Drive Anda
5. Download model dan copy ke folder `src/models/`

**Waktu Training:**
- BiLSTM: ~5-10 menit
- BERT: ~30-45 menit (dengan GPU)
- DistilBERT: ~20-30 menit (dengan GPU)

---

### 💻 Menjalankan Dashboard Streamlit (Lokal)

#### 1️⃣ Clone Repository

```bash
git clone https://github.com/RNikmah/SMS_Spam_Prediction_UAP-ML.git
cd SMS_Spam_Prediction_UAP-ML
```

#### 2️⃣ Install Dependencies

Disarankan menggunakan virtual environment.

```bash
pip install -r requirements.txt
```

#### 3️⃣ Struktur Folder

```
├── Spam_Detection.ipynb        # Notebook training di Google Colab
├── README.md
├── pyproject.toml
└── src/
    ├── app.py                  # Aplikasi Streamlit
    ├── requirements.txt
    ├── models/                 # Model yang sudah di-training
│   │   ├── bilstm/
│   │   │   ├── bilstm_spam_model.keras
│   │   │   ├── bilstm_tokenizer.pkl
│   │   │   └── bilstm_config.json
│   │   ├── bert/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── tokenizer.json
│   │   │   └── vocab.txt
│   │   └── distilbert/
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   ├── tokenizer.json
│       │   └── vocab.txt
│   ├── assets/
│   │   ├── Plot Acc BiLSTM.png
│   │   ├── Plot Loss BiLSTM.png
│   │   ├── Confusion Matrix BiLSTM.png
│   │   ├── Plot Acc BERT.png
│   │   ├── Plot Loss BERT.png
│   │   ├── Confusion Matrix BERT.png
│   │   ├── Plot Acc DistilBERT.png
│   │   ├── Plot Loss DistilBERT.png
│   │   └── Confusion Matrix DistilBERT.png
│   └── inference/
│       ├── bert_infer.py
│       ├── bilstm_infer.py
│       └── distilbert_infer.py
```

#### 4️⃣ Pastikan Model Tersedia

**Model BiLSTM** (lokal):
- Pastikan folder `src/models/bilstm/` berisi:
  - `bilstm_spam_model.keras`
  - `bilstm_tokenizer.pkl`
  - `bilstm_config.json`

**Model BERT & DistilBERT** (Hugging Face Hub):
- Model di-load otomatis dari Hugging Face Hub
- Repository:
  - BERT: [`Rahma13/spam-detection-bert`](https://huggingface.co/Rahma13/spam-detection-bert)
  - DistilBERT: [`Rahma13/spam-detection-distilbert`](https://huggingface.co/Rahma13/spam-detection-distilbert)
- Tidak perlu download manual, aplikasi akan download otomatis saat pertama kali dijalankan

#### 5️⃣ Jalankan Streamlit

Pastikan folder `src/models/` berisi:
- `bilstm/` - Model BiLSTM dan tokenizer
- `bert/` - Model BERT fine-tuned
- `distilbert/` - Model DistilBERT fine-tuned

Jika model belum ada, lakukan training di Google Colab terlebih dahulu (lihat panduan training di atas).

#### 5️⃣ Jalankan Streamlit

```bash
cd src
streamlit run app.py
```

#### 6️⃣ Akses Aplikasi

Buka browser dan akses:

```
http://localhost:8501
```

---

## 📸 Fitur Aplikasi

### 🔍 Halaman Deteksi
- **Input Area**: Text area untuk memasukkan pesan SMS/Email
- **Model Selection**: Dropdown di sidebar untuk memilih model (BiLSTM/BERT/DistilBERT)
- **Threshold Control**: Slider untuk mengatur threshold spam detection
- **Result Display**: Badge gradient menampilkan hasil prediksi (Spam/Ham)
- **Confidence Visualization**: Progress bar dan metric cards
- **History Tracking**: Riwayat prediksi dalam expander dengan tombol clear

### 🧠 Halaman Info Model
- **Penjelasan Detail**: Deskripsi lengkap setiap model
- **Karakteristik**: Kelebihan dan kekurangan
- **Arsitektur**: Struktur layer model
- **Use Case**: Kapan sebaiknya menggunakan model tertentu

### 📊 Halaman Evaluasi Model
- **Training Plots**: Grafik accuracy dan loss per epoch
- **Confusion Matrix**: Visualisasi klasifikasi hasil test set
- **Metrics Explanation**: Penjelasan metrik evaluasi
- **Comparison**: Perbandingan performa antar model

---

## 📦 Dependencies

### Untuk Dashboard (Streamlit)

```
streamlit>=1.28.0
tensorflow>=2.15.0
torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

### Untuk Training (Google Colab)

```
datasets
acceleratetransformers
torch
tensorflow
matplotlib
seaborn
scikit-learn
```

**Catatan:** Google Colab sudah menyediakan sebagian besar library. Install tambahan jika diperlukan.

---

## 📌 Catatan Tambahan

- ✅ Seluruh training dilakukan di **Google Colab** dengan GPU support
- ✅ Evaluasi model dilakukan pada **test set** yang tidak pernah dilihat saat training
- ✅ Model pretrained BERT & DistilBERT diambil dari **Hugging Face** (`bert-base-uncased` dan `distilbert-base-uncased`)
- ✅ **Model fine-tuned** di-host di Hugging Face Hub:
  - [`Rahma13/spam-detection-bert`](https://huggingface.co/Rahma13/spam-detection-bert)
  - [`Rahma13/spam-detection-distilbert`](https://huggingface.co/Rahma13/spam-detection-distilbert)
- ✅ BiLSTM di-training from scratch dengan Keras/TensorFlow (tersimpan lokal)
- ✅ Dataset split: **70% training**, **15% validation**, **15% test** (stratified)
- ✅ Model BiLSTM memerlukan `bilstm_tokenizer.pkl` yang di-generate saat training
- ✅ DistilBERT menggunakan **weighted loss** untuk menangani class imbalance
- ✅ Sistem dirancang untuk kebutuhan **akademik dan demonstrasi**
- ⚠️ **Tidak disarankan** untuk production tanpa fine-tuning lebih lanjut

---

## 👨‍💻 Author

**Rahmatun Nikmah - 202210370311109**

---

## 📄 License

Project ini dibuat untuk menyelesaikan Praktikum *Machine Learning*.

---

## 🔗 Links

- [Hugging Face - BERT](https://huggingface.co/bert-base-uncased)
- [Hugging Face - DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [Streamlit Documentation](https://docs.streamlit.io/)
