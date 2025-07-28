import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

label_map = {0: "yes", 1: "no", 2: "maybe"}
model_path = "models/scibert-classifier/checkpoint-2500"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

st.title("MedIntel - Biomedical QA Classifier")
st.markdown("This AI model predicts **yes/no/maybe** based on a question and biomedical abstract.")

question = st.text_input("Enter your biomedical question:")
context = st.text_area("Paste abstract or medical context here:", height=200)

if st.button("Classify"):
    if question.strip() == "" or context.strip() == "":
        st.warning("Please enter both question and context.")
    else:
        text = question + " [SEP] " + context
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = label_map[pred_idx]
            confidence = probs[0][pred_idx].item()

        st.success(f"Prediction: **{pred_label.upper()}** (Confidence: {confidence:.2f})")