from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rag.search_pubmed import search_passages
import torch
import torch.nn.functional as F

# Load your fine-tuned model
checkpoint = "models/scibert-classifier/checkpoint-2500"
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

labels = ["yes", "no", "maybe"]

while True:
    question = input("Ask a biomedical question (or 'exit'): ")
    if question.lower() == "exit":
        break

    # Step 1: Retrieve relevant context
    passages = search_passages(question)
    full_context = " ".join(passages)

    # Step 2: Run through SciBERT classifier
    input_text = question + " [SEP] " + full_context
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

    predicted = labels[probs.argmax().item()]
    print(f"\n🧠 Predicted Answer: {predicted} (confidence: {probs.max():.2f})\n")
