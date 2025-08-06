# 📌 RAG System for IMF Reports

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to extract **key numbers, statistics, and trends** from IMF reports.  
It integrates:
- **Web Crawling** of IMF content (HTML, PDFs, tables)
- **Data Cleaning** (remove boilerplate, non-English text)
- **Hybrid Search** (BM25 + Vector Search) with Cross-Encoder reranking
- **Local LLM (Llama 3.1 8B via Ollama)** for high-quality answers.

---

## 🚀 Pipeline Overview

The RAG system follows these main stages:

---

### **1️⃣ Data Collection (Web Crawler + PDF Loader + Table Extractor)**

We automatically collect IMF content from the official site:

- **`web_crawler.py`** crawls IMF pages and detects:
  - HTML pages (text)
  - PDF reports (linked in the site)
- **`pdf_loader.py`** downloads and extracts:
  - Full PDF text
  - Tables (CSV format) from PDFs
- **`table_extractor.py`** extracts tables embedded in HTML

Run ingestion:
```bash
python ingest_pipeline.py --start-url "https://www.imf.org/en/Publications" --max-pages 50
```
Outputs:
- `data/output/collected_data.json` — combined text + tables
- `data/output/data_log.csv` — provenance log
- `data/output/failed_links.csv` — failed fetch attempts
- `data/pdf_texts/` — extracted PDF text files
- `data/pdf_tables/` — extracted PDF tables (CSV)

---

### **2️⃣ Data Cleaning (Boilerplate & Language Filtering)**

After ingestion, remove **boilerplate phrases** and **non-English entries**:

- **`clean_json_for_rag.py`**:
  - Detects language (`langdetect`)
  - Removes common IMF boilerplate patterns
  - Outputs a cleaned dataset for embeddings

Run cleaning:
```bash
python clean_json_for_rag.py
```

Outputs:
- `data/output/collected_data_clean.json` — cleaned, English-only dataset

---

### **3️⃣ Targeted Semantic Chunking**
Chunks cleaned text into **semantic segments** to preserve numbers, statistics, and context.
```bash
python run_graph.py --build
```
Outputs:
- Targeted chunks ready for embedding and storage in Qdrant

---

### **4️⃣ Hybrid Search (BM25 + Vector Search)**
Retrieves documents using:
- **BM25** (keyword search from Qdrant payload)
- **Vector Search** (semantic similarity)
- **Cross-Encoder Reranker** (precision improvement)
```bash
python run_graph.py --query "Extract IMF's latest numbers and trends"
```

---

### **5️⃣ Query the LLM (Llama 3.1 8B via Ollama)**
Pass ranked context to local LLM:
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
Expose the RAG system as an API:
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
│   ├── output/
│   ├── pdf_tables/
│   ├── pdf_texts/
│   └── pdfs/
│── src/
│   ├── ingestion/
│   │   ├── web_crawler.py
│   │   ├── pdf_loader.py
│   │   ├── table_extractor.py
│   │   └── ingest_pipeline.py
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
│── clean_json_for_rag.py
│── run_graph.py
│── requirements.txt
│── Dockerfile
│── README.md
```

---

## 🔑 Key Components
- **Web Scraper** (HTML, PDFs, Tables)
- **Data Cleaner** (Language filter + boilerplate removal)
- **Qdrant** — Vector DB for document storage + BM25
- **Hybrid Retriever** — Semantic + lexical search
- **Cross-Encoder Reranker** — Improves ranking
- **Ollama + Llama 3.1** — Local LLM inference
