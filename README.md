#  MedIntel – Biomedical QA Chatbot (SciBERT + RAG)

**MedIntel** is a production-grade AI assistant that answers biomedical questions using a fine-tuned SciBERT model and Retrieval-Augmented Generation (RAG) pipeline over PubMed literature.

##  Demo
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Live--Demo-red?logo=streamlit)](http://localhost:8501)

##  Model Highlights

-  Fine-tuned SciBERT for biomedical question classification (`yes`/`no`/`maybe`)
-  Retrieval-Augmented Generation (RAG) using FAISS + MiniLM + PubMed
-  Streamlit frontend for real-time interactive QA
-  Dockerized and production-ready
-  Clean architecture and modular code

---

##  Project Structure


##  How to Run

### 1. Install Requirements

```bash

pip install -r requirements.txt

2. Run the App (Local)
streamlit run streamlit_app/app.py

3. Run with Docker
docker build -t medintel-app .
docker run -p 8501:8501 medintel-app


python train_classifier.py


python rag/index_pubmed.py


Author
Chetan Nani
Built with 💡 to showcase end-to-end AI product engineering in healthcare.


---

Once this is saved and committed, reply `done`, and I’ll be moving to **cloud deployment (Render or Hugging Face)** so you can showcase it instantly with a public link.

