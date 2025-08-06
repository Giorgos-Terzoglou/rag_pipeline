# 📌 RAG System for IMF Reports

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to extract **key numbers, statistics, and trends** from IMF reports. It integrates **Hybrid Search (BM25 + Vector Search)** with **Cross-Encoder reranking** and a **local LLM (Llama 3.1 8B via Ollama)** for high-quality answers.

---

## 🚀 Pipeline Overview

The RAG system follows these main stages:

### **1️⃣ Data Collection & Cleaning**
- Ingest IMF-related content (reports, factsheets, statistics)
- Clean and normalize data into structured JSON
```bash
python ingest_pipeline.py --source ./data/input/imf_data.json
```
Outputs:
- `data/output/cleaned_data.json` — Cleaned & normalized IMF dataset

---

### **2️⃣ Targeted Semantic Chunking**
- Chunk text into **semantic segments** using `SemanticChunker` to capture context-rich passages (numbers, tables, trends)
```bash
python run_graph.py --build
```
Outputs:
- `1459 targeted chunks` created from reports

---

### **3️⃣ Embedding Generation**
- Embeddings generated for each chunk using:
  - **SentenceTransformers (all-MiniLM-L6-v2)** *(local, free)*
  - (Optional) **OpenAI text-embedding-3-small** *(cloud, paid)*
```python
vectors = embed_documents(chunks, use_openai=False)
```
Outputs:
- `1459 vectors (dim=384)` stored with metadata

---

### **4️⃣ Vector Search Test (FAISS)**
- Before uploading to Qdrant, validate embeddings locally:
```python
results = similarity_search(query="What is the IMF World Economic Outlook?", k=3)
```
Ensures embeddings retrieve relevant chunks.

---

### **5️⃣ Upload to Qdrant**
- Store vectors & metadata in Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
python upload_to_qdrant.py
```
Collection: `rag_collection`

---

### **6️⃣ Hybrid Search (BM25 + Vector Search)**
- Retrieve documents using **Hybrid Retriever**:
  - **BM25**: Keyword-based search from Qdrant payload
  - **Vector Search**: Semantic similarity from embeddings
  - **Cross-Encoder Reranker**: Reorders top results for precision
```python
candidates = hybrid_retriever.search(question, k=15)
ranked = reranker.rerank(question, candidates, top_k=5)
```

---

### **7️⃣ Query the LLM (Llama 3.1 8B via Ollama)**
- Ranked context passed to local LLM for final answer:
```bash
ollama pull llama3.1:8b
python run_graph.py --query "Extract IMF's latest numbers and trends"
```

Example Output:
```
Numbers:
- 191 member countries
- $1 trillion lending capacity
- 9 (11%) actions completed
- 34 (42%) actions underway

Trends:
- Moderate progress on IMF initiatives
- Focus on financial stability and trade growth
```

---

## 📡 API Access (FastAPI)

The RAG system is also exposed via an API for production use:

Run API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Swagger Docs:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🏗 Project Structure
```
project/
│── data/
│   ├── input/
│   └── output/
│── src/
│   ├── graph/
│   │   ├── build_graph.py
│   │   └── nodes.py
│   ├── retrieval/
│   │   ├── bm25_search.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   └── llm/
│       └── rag_pipeline.py
│── api/
│   └── main.py
│── run_graph.py
│── requirements.txt
│── Dockerfile
│── README.md
```

---

## 🔑 Key Components
- **Qdrant** — Vector DB for document storage and BM25 text payload
- **BM25** — Keyword search from Qdrant payload
- **FAISS** — Local similarity search for pre-upload testing
- **Hybrid Retriever** — Combines semantic + lexical retrieval
- **Cross-Encoder Reranker** — Reorders retrieved documents
- **Ollama + Llama 3.1** — Fast local inference using LLMs
