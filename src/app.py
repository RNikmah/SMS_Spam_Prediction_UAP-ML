import streamlit as st
import numpy as np
import torch
import pickle
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="wide"
)

# ======================================================
# PATH CONFIGURATION
# ======================================================
# Dapatkan directory dari file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path untuk BiLSTM model
BILSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "bilstm_spam_model.keras")
BILSTM_TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "bilstm_tokenizer.pkl")

# Path untuk assets
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
.badge {
    padding: 8px 16px;
    border-radius: 12px;
    font-weight: bold;
    color: white;
    font-size: 1.1em;
    display: inline-block;
    margin: 10px 0;
}
.spam { 
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    box-shadow: 0 4px 6px rgba(231, 76, 60, 0.3);
}
.ham { 
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    box-shadow: 0 4px 6px rgba(46, 204, 113, 0.3);
}
.main-header {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    margin-bottom: 30px;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_bilstm():
    """Load BiLSTM model dan tokenizer dari file lokal"""
    try:
        # Debug: Print paths untuk troubleshooting
        print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Looking for model at: {BILSTM_MODEL_PATH}")
        print(f"[DEBUG] Model exists: {os.path.exists(BILSTM_MODEL_PATH)}")
        
        if not os.path.exists(BILSTM_MODEL_PATH):
            st.error(f"❌ Model BiLSTM tidak ditemukan di: {BILSTM_MODEL_PATH}")
            st.info(f"Current working directory: {os.getcwd()}")
            st.info(f"BASE_DIR: {BASE_DIR}")
            return None, None
        
        model = load_model(BILSTM_MODEL_PATH)
        
        if not os.path.exists(BILSTM_TOKENIZER_PATH):
            st.error(f"❌ Tokenizer BiLSTM tidak ditemukan di: {BILSTM_TOKENIZER_PATH}")
            return None, None
            
        with open(BILSTM_TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        
        print("[DEBUG] BiLSTM model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading BiLSTM: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

@st.cache_resource
def load_transformer(model_choice):
    """Load BERT/DistilBERT dari Hugging Face Hub"""
    try:
        if model_choice == "BERT":
            repo_id = "Rahma13/spam-detection-bert"
        else:  # DistilBERT
            repo_id = "Rahma13/spam-detection-distilbert"
        
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        model.eval()
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading {model_choice}: {str(e)}")
        return None, None

# ======================================================
# PREDICTION FUNCTION
# ======================================================
def predict(text, model_choice):
    """Fungsi prediksi untuk semua model"""
    
    if model_choice == "BiLSTM":
        model, tokenizer = load_bilstm()
        
        if model is None or tokenizer is None:
            st.error("❌ BiLSTM model gagal dimuat. Silakan gunakan model lain.")
            return "Error", 0.0
        
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=100, padding="post")
        prob = model.predict(pad, verbose=0)[0][0]
        label = "Spam" if prob > 0.5 else "Ham"
        confidence = float(prob) if label == "Spam" else float(1 - prob)

    elif model_choice == "BERT":
        tokenizer, model = load_transformer("BERT")
        
        if model is None or tokenizer is None:
            st.error("❌ BERT model gagal dimuat.")
            return "Error", 0.0
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)
        label = ["Ham", "Spam"][idx]
        confidence = float(confidence)

    else:  # DistilBERT
        tokenizer, model = load_transformer("DistilBERT")
        
        if model is None or tokenizer is None:
            st.error("❌ DistilBERT model gagal dimuat.")
            return "Error", 0.0
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)
        label = ["Ham", "Spam"][idx]
        confidence = float(confidence)

    return label, confidence

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📱 Navigasi")

page = st.sidebar.radio(
    "Pilih Halaman",
    ["🔍 Deteksi", "🧠 Info Model", "📊 Evaluasi Model"]
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Pengaturan")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["BERT", "DistilBERT", "BiLSTM"]  # BERT & DistilBERT lebih stabil
)

threshold = st.sidebar.slider(
    "Threshold Spam",
    0.0, 1.0, 0.5
)

# ======================================================
# MAIN UI
# ======================================================

# ============== PAGE: DETEKSI ==============
if page == "🔍 Deteksi":
    st.markdown("""
    <div class='main-header'>
        <h1>📩 Spam Message Detector</h1>
        <p>UAP ML Project</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([2, 1])

    with col_input:
        text_input = st.text_area(
            "Masukkan teks SMS / Email:",
            height=150,
            placeholder="Contoh: Congratulations! You've won $1000. Click here to claim..."
        )
        
        predict_btn = st.button("🚀 Prediksi", use_container_width=True, type="primary")

    with col_result:
        st.markdown("### 🎯 Hasil Prediksi")
        result_container = st.container()

    # ======================================================
    # SESSION STATE
    # ======================================================
    if "history" not in st.session_state:
        st.session_state.history = []

    # ======================================================
    # PREDICT BUTTON
    # ======================================================
    if predict_btn:
        if text_input.strip() == "":
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            with st.spinner(f"🔍 Menganalisis dengan {model_choice}..."):
                label, confidence = predict(text_input, model_choice)
                
                if label == "Error":
                    st.error("❌ Prediksi gagal. Silakan coba model lain.")
                else:
                    final_label = (
                        "Spam" if (label == "Spam" and confidence >= threshold)
                        else "Ham"
                    )

                badge_class = "spam" if final_label == "Spam" else "ham"

                    with result_container:
                        st.markdown(
                            f"<span class='badge {badge_class}'>{final_label}</span>",
                            unsafe_allow_html=True
                        )

                        st.progress(confidence)
                        
                        col_metric1, col_metric2 = st.columns(2)
                        col_metric1.metric("Confidence", f"{confidence:.2%}")
                        col_metric2.metric("Model", model_choice)

                        st.session_state.history.insert(0, {
                        "Text": text_input[:40] + "..." if len(text_input) > 40 else text_input,
                        "Model": model_choice,
                        "Prediction": final_label,
                        "Confidence": round(confidence, 3)
                    })
                    
                    st.success("✅ Prediksi berhasil!")

    # ======================================================
    # HISTORY
    # ======================================================
    st.markdown("---")
    with st.expander("📜 Riwayat Prediksi", expanded=False):
        if st.session_state.history:
            st.dataframe(
                st.session_state.history,
                use_container_width=True,
                hide_index=True
            )
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Belum ada riwayat prediksi.")

# ============== PAGE: INFO MODEL ==============
elif page == "🧠 Info Model":
    st.markdown("""
    <div class='main-header'>
        <h1>🧠 Informasi Model</h1>
        <p>Penjelasan arsitektur dan karakteristik model</p>
    </div>
    """, unsafe_allow_html=True)

    if model_choice == "BiLSTM":
        st.markdown("""
        ### 🔹 BiLSTM (Baseline)
        
        **Bidirectional Long Short-Term Memory** adalah model neural network berbasis RNN yang membaca teks dari dua arah (kiri-ke-kanan dan kanan-ke-kiri).
        
        #### Karakteristik:
        - ✅ Neural Network non-pretrained
        - ✅ Cepat & ringan
        - ✅ Cocok sebagai baseline
        - ✅ Training dari scratch dengan dataset spam
        - ✅ Sequence length: 100 tokens
        
        #### Arsitektur:
        - Embedding Layer
        - Bidirectional LSTM
        - Dense Layer
        - Sigmoid Output (Binary Classification)
        """)
        
    elif model_choice == "BERT":
        st.markdown("""
        ### 🔹 BERT (Bidirectional Encoder Representations from Transformers)
        
        **BERT** adalah model transformer pretrained yang dikembangkan Google. Model ini menggunakan attention mechanism untuk memahami konteks kata dalam kalimat.
        
        #### Karakteristik:
        - ✅ Transformer pretrained
        - ✅ Contextual embedding dua arah
        - ✅ Akurasi tinggi
        - ✅ Fine-tuned untuk spam detection
        - ✅ WordPiece tokenization
        
        #### Kelebihan:
        - Memahami konteks kata secara mendalam
        - Transfer learning dari pretrained model
        - State-of-the-art performance
        """)
        
    else:
        st.markdown("""
        ### 🔹 DistilBERT (Distilled BERT)
        
        **DistilBERT** adalah versi ringkas dari BERT yang menggunakan knowledge distillation untuk mengkompresi model tanpa kehilangan performa signifikan.
        
        #### Karakteristik:
        - ✅ Versi ringkas BERT (40% lebih kecil)
        - ✅ Lebih cepat (60% faster)
        - ✅ Performa mendekati BERT (97% retained)
        - ✅ Hemat resource dan komputasi
        
        #### Keuntungan:
        - Inference lebih cepat
        - Memory footprint lebih kecil
        - Cocok untuk production deployment
        - Tetap mempertahankan akurasi tinggi
        """)

# ============== PAGE: EVALUASI MODEL ==============
elif page == "📊 Evaluasi Model":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Evaluasi Model</h1>
        <p>Metrik performa dan visualisasi hasil training</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(f"Model: {model_choice}")
    
    col1, col2, col3 = st.columns(3)

    # Gunakan path yang benar untuk assets
    try:
        if model_choice == "BiLSTM":
            col1.image(os.path.join(ASSETS_DIR, "Plot Acc BiLSTM.png"), caption="Plot Accuracy", use_container_width=True)
            col2.image(os.path.join(ASSETS_DIR, "Plot Loss BiLSTM.png"), caption="Plot Loss", use_container_width=True)
            col3.image(os.path.join(ASSETS_DIR, "Confusion Matrix BiLSTM.png"), caption="Confusion Matrix", use_container_width=True)

        elif model_choice == "BERT":
            col1.image(os.path.join(ASSETS_DIR, "Plot Acc BERT.png"), caption="Plot Accuracy", use_container_width=True)
            col2.image(os.path.join(ASSETS_DIR, "Plot Loss BERT.png"), caption="Plot Loss", use_container_width=True)
            col3.image(os.path.join(ASSETS_DIR, "Confusion Matrix BERT.png"), caption="Confusion Matrix", use_container_width=True)

        else:
            col1.image(os.path.join(ASSETS_DIR, "Plot Acc DistilBERT.png"), caption="Plot Accuracy", use_container_width=True)
            col2.image(os.path.join(ASSETS_DIR, "Plot Loss DistilBERT.png"), caption="Plot Loss", use_container_width=True)
            col3.image(os.path.join(ASSETS_DIR, "Confusion Matrix DistilBERT.png"), caption="Confusion Matrix", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading images: {str(e)}")
        st.info(f"📂 Assets directory: {ASSETS_DIR}")

    st.markdown("---")
    st.caption("📌 Evaluasi dilakukan pada test set secara offline.")
    
    st.markdown("""
    ### 📋 Penjelasan Metrik:
    
    - **Accuracy**: Persentase prediksi yang benar dari total prediksi
    - **Loss**: Fungsi loss yang mengukur error model (semakin rendah semakin baik)
    - **Confusion Matrix**: Visualisasi performa klasifikasi yang menunjukkan True Positive, True Negative, False Positive, dan False Negative
    """)
