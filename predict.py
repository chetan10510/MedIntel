from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys

label_map = {0: "yes", 1: "no", 2: "maybe"}

model_path = "models/scibert-classifier/checkpoint-2500"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(question, context):
    text = question + " [SEP] " + context
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        return label_map[pred_label], probs[0][pred_label].item()

if __name__ == "__main__":
    q = input("Enter question: ")
    c = input("Enter context: ")
    label, confidence = predict(q, c)
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
