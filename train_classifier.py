import json
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load data
with open("data/processed/qa_dataset.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f]

df = pd.DataFrame(lines)
df["text"] = df["question"].astype(str) + " [SEP] " + df["context"].astype(str)


label2id = {"yes": 0, "no": 1, "maybe": 2}
df["label"] = df["label"].map(label2id)

# Tokenizer and dataset
tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Split
train_test = dataset.train_test_split(test_size=0.2)
train_ds = train_test["train"]
eval_ds = train_test["test"]

# Model
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=3)

# Training config
training_args = TrainingArguments(
    output_dir="models/scibert-classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()
trainer.save_model("models/scibert-classifier")
print("Training complete and model saved to models/scibert-classifier")
