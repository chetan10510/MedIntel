import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss

DATA_FILE = "data/processed/pubmed_passages.jsonl"
INDEX_FILE = "rag/faiss_index.index"
META_FILE = "rag/passages.pkl"

os.makedirs("rag", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


print("🔍 Loading passages...")

texts = []
metas = []

with open(DATA_FILE, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ctx_obj = obj.get("context", "")
        if isinstance(ctx_obj, dict) and "contexts" in ctx_obj:
            context_text = " ".join(ctx_obj["contexts"])  # join paragraphs
        elif isinstance(ctx_obj, str):
            context_text = ctx_obj
        else:
            context_text = ""

        if len(context_text) > 100:
            chunks = [context_text[i:i+500] for i in range(0, len(context_text), 500)]
            for chunk in chunks:
                texts.append(chunk)
                metas.append({
                    "source": obj["question"][:100] + "..." if "question" in obj else "abstract"
                })

print(f"✅ Total chunks: {len(texts)}")

if not texts:
    raise ValueError("❌ No valid passages found. Check your input file or field names.")

print("🧠 Generating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

print("💾 Saving FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(metas, f)

print(f"✅ Indexed {len(texts)} chunks to {INDEX_FILE}")
