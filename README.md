# 📌 RAG System for IMF Reports

This project implements a **Retrieval-Augmented Generation (RAG)** system that scrapes IMF data, chunks it, embeds it, stores it in **Qdrant**, and retrieves it using **Hybrid Search (BM25 + Vector Search)** with reranking before passing context to an **LLM (Llama 3.1 8B via Ollama)**.

---

## 🚀 Pipeline Overview
The pipeline consists of the following steps:

### 1️⃣ Data Collection & Cleaning
We ingest IMF-related content (documents, reports, factsheets) and clean it.
```bash
python ingest_pipeline.py --source ./data/input/imf_data.json
```
Outputs:
- `data/output/cleaned_data.json` — cleaned and normalized JSON data

### 2️⃣ Targeted Semantic Chunking
We chunk IMF documents using **targeted semantic segmentation** to capture meaningful sentences (e.g., numbers, trends, tables) instead of fixed-size tokens.
```bash
python run_graph.py --build
```
Outputs:
- `1459 targeted chunks` created from documents

### 3️⃣ Embedding Generation
We generate embeddings for each chunk using:
- **SentenceTransformers (all-MiniLM-L6-v2)** *(local, free)*
- (Optional) **OpenAI text-embedding-3-small** *(cloud, paid)*
```python
vectors = embed_documents(chunks, use_openai=False)
```
Outputs:
- `1459 vectors of dimension 384`

### 4️⃣ Local Similarity Search Test (FAISS)
Before uploading to Qdrant, test embeddings locally:
```python
results = similarity_search(query="What is the IMF World Economic Outlook?", k=3)
```
This ensures embeddings are relevant.

### 5️⃣ Upload to Qdrant
Upload vectors and metadata to **Qdrant**:
```python
upload_to_qdrant(vectors, docs, collection_name="rag_collection")
```
Run Qdrant locally:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 6️⃣ Retrieval (Hybrid Search: Vector + BM25 from Qdrant)
We retrieve relevant documents using **Hybrid Retrieval**:
- **BM25**: Keyword-based scoring (from Qdrant text payload)
- **Vector Search**: Semantic similarity via embeddings
- **Reranking**: Cross-encoder reorders top candidates for better accuracy
```python
candidates = hybrid_retriever.search(question, k=15)
ranked = reranker.rerank(question, candidates, top_k=5)
```

### 7️⃣ Query the LLM (Llama 3.1 8B via Ollama)
The ranked context is passed to the LLM:
```python
ollama pull llama3.1:8b
python run_graph.py --query "Extract IMF's latest numbers and trends"
```
The retrieved context is sent to the LLM along with the user question.

---

## 📡 Example Usage
### Build Qdrant Index
```bash
python run_graph.py --build
```

### Query the RAG System
```bash
python run_graph.py --query "Extract IMF's latest numbers and trends"
```

Example output:
```
Numbers:
- 191 member countries
- $1 trillion lending capacity
- 9 (11%) actions completed
- 34 (42%) actions underway

Trends:
- Moderate progress on IMF initiatives
- Significant focus on financial stability and trade growth
```

---

## 📌 Next Steps for Production-Ready RAG System
- ✅ Improve **chunking** to better capture numbers & trends
- ✅ Add **Hybrid Search** (BM25 from Qdrant + FAISS)
- 🔄 Add **evaluation scripts** to measure retrieval accuracy
- 🔄 Implement **prompt templates** for structured responses (tables, JSON)
- 🔄 Create **FastAPI endpoint** for external queries

---

## 🏗 Project Structure
```
project/
│── data/
│   │── input/
│   │── output/
│── src/
│   │── graph/
│   │   │── build_graph.py
│   │   │── nodes.py
│   │── retrieval/
│   │   │── bm25_search.py
│   │   │── hybrid_search.py
│   │   │── reranker.py
│   │── llm/
│   │   │── rag_pipeline.py
│── run_graph.py
│── README.md
```

---

## 🔑 Key Components
- **Qdrant** — Vector database for embeddings & metadata
- **BM25** — Keyword search from Qdrant payload
- **FAISS** — Local similarity search for testing
- **Hybrid Retriever** — Combines BM25 & vector search
- **Reranker** — Improves retrieval precision
- **Llama 3.1 8B** — Local LLM for answer generation via Ollama
