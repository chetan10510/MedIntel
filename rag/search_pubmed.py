import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_FILE = "rag/faiss_index.index"
META_FILE = "rag/passages.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    metas = pickle.load(f)

def search_passages(query, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(metas):
            text = metas[idx].get("source", "")
            results.append(text)
    return results
