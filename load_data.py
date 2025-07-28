from datasets import load_dataset
import json
import os

os.makedirs("data/processed", exist_ok=True)

print("Loading PubMedQA...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

records = []

for row in dataset:
    if row["final_decision"] in ["yes", "no", "maybe"]:
        question = row["question"]
        context = row.get("context", "") or row.get("long_answer", "") or ""
        records.append({
            "question": question,
            "context": context,
            "label": row["final_decision"]
        })

with open("data/processed/qa_dataset.jsonl", "w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row) + "\n")

print(f"Saved {len(records)} records to data/processed/qa_dataset.jsonl")
