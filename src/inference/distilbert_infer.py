import torch
import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

@st.cache_resource
def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    model.eval()
    return tokenizer, model

def predict_distilbert(text):
    tokenizer, model = load_distilbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    spam_prob = probs[0][1].item()
    label = "Spam" if spam_prob > 0.5 else "Ham"

    return label, spam_prob * 100
