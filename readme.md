# RAG Pipeline for IMF Reports

The pipeline consists of the following steps:

---

### 1️⃣ Data Ingestion
We ingest cleaned IMF reports stored in JSON format.
```bash
python run_graph.py --build
```
This outputs:
- `data/output/cleaned_data.json` — cleaned normalized JSON

---

### 2️⃣ Targeted Semantic Chunking
Chunks are created using **targeted chunking logic** (splitting by report sections, keeping numbers and trends intact).
```python
chunks = targeted_chunking(clean_data)
```
This ensures better retrieval for IMF-specific queries.

---

### 3️⃣ Embedding Generation
We generate embeddings using **SentenceTransformers (all-MiniLM-L6-v2)** *(local, free)*.
```python
vectors = embed_documents(chunks)
```
Each chunk is converted into a 384-dimensional vector.

---

### 4️⃣ Similarity Search (Local Test)
Before uploading to Qdrant, we test similarity search locally using **FAISS**:
```python
results = similarity_search(query="What is the IMF World Economic Outlook?", k=3)
```
This validates embedding quality before indexing.

---

### 5️⃣ Upload to Qdrant
We upload embeddings and metadata to **Qdrant** for persistent storage.
```python
upload_to_qdrant(vectors, docs, collection_name="rag_collection")
```
Run Qdrant locally:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

### 6️⃣ Hybrid Retrieval (Semantic + BM25)
At query time, we:
- Retrieve semantic matches from Qdrant
- Retrieve keyword matches from BM25 (stored in Qdrant payloads)
- Merge and rerank results
```python
hybrid_results = hybrid_search(query="Extract IMF latest numbers and trends")
```

---

### 7️⃣ LLM Query with Local Ollama
Context is sent to **Llama 3.1 8B** running locally via Ollama.
```python
ollama run llama3.1:8b
```
Pipeline query example:
```bash
python run_graph.py --query "Extract IMF's latest numbers and trends"
```

---

### 📂 Example Project Structure
```plaintext
rag_project/
│── data/
│   ├── output/
│   │   ├── cleaned_data.json
│── src/
│   ├── graph/
│   │   ├── build_graph.py
│   │   ├── nodes.py
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   ├── bm25_search.py
│   │   ├── reranker.py
│   ├── llm/
│   │   ├── rag_pipeline.py
│── run_graph.py
