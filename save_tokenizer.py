from transformers import AutoTokenizer

model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("models/scibert-classifier/checkpoint-2500")
