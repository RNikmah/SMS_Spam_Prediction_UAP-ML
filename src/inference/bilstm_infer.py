import pickle, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

@st.cache_resource
def load_bilstm():
    model = load_model("models/bilstm/bilstm_spam_model.keras")

    with open("models/bilstm/bilstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("models/bilstm/bilstm_config.json") as f:
        config = json.load(f)

    return model, tokenizer, config

def predict_bilstm(text):
    model, tokenizer, config = load_bilstm()

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=config["max_len"], padding="post")

    prob = model.predict(pad)[0][0]
    label = "Spam" if prob > 0.5 else "Ham"

    return label, prob * 100
