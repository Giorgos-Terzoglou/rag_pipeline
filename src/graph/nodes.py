import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import spacy
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.vectorstore.qdrant_setup import get_qdrant_client
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

def test_similarity_search(docs, vectors, use_openai=True):
    print("🔍 Testing similarity search locally (FAISS)...")

    if use_openai:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in docs]
    faiss_store = FAISS.from_texts(texts, embeddings_model)

    query = "global economic outlook"
    results = faiss_store.similarity_search(query, k=3)

    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}:\nSource: {r.metadata.get('source', 'N/A')}\nContent: {r.page_content[:300]}...")
    return results


# -------- Node 1: Load Clean Data --------
def load_clean_data():
    print("📂 Loading cleaned JSON...")
    with open("data/output/collected_data_clean.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} documents")
    return data

# -------- Node 2: Chunk Text --------
def chunk_documents(data):
    print("🧠 Performing targeted chunking for IMF reports...")
    nlp = spacy.load("en_core_web_trf")

    docs = []
    for d in data:
        text = d["content"]
        meta = {"source": d["source"]}

        # Split on sections, not just sentences
        sections = re.split(r"(World Economic Outlook|Fiscal Monitor|Global Financial Stability Report|Annex \d+|Table \d+)", text)
        for sec in sections:
            if len(sec.strip()) < 50:
                continue
            doc_spacy = nlp(sec)
            chunk = []
            for sent in doc_spacy.sents:
                chunk.append(sent.text)
                if sum(len(s) for s in chunk) > 1200:
                    docs.append(Document(page_content=" ".join(chunk), metadata=meta))
                    chunk = []
            if chunk:
                docs.append(Document(page_content=" ".join(chunk), metadata=meta))

    print(f"✅ Created {len(docs)} targeted chunks")
    return docs


# -------- Node 3: Embed Documents --------
def embed_documents(docs, use_openai=True):
    if use_openai:
        print("🧠 Creating embeddings using OpenAI...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectors = [embeddings.embed_query(doc.page_content) for doc in docs]
    
    else:
        try:
            print("🧠 Creating embeddings using SentenceTransformers (all-MiniLM-L6-v2)...")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vectors = model.encode([doc.page_content for doc in docs], convert_to_numpy=True).tolist()
        
        except Exception as e:
            print(f"⚠️ SentenceTransformers failed: {e}")
            print("⚠️ Falling back to spaCy medium vectors...")
            
            try:
                nlp = spacy.load("en_core_web_md")
            except OSError:
                print("⚠️ Model 'en_core_web_md' not found. Installing...")
                os.system("python -m spacy download en_core_web_md")
                nlp = spacy.load("en_core_web_md")

            vectors = []
            for doc in docs:
                vec = nlp(doc.page_content).vector
                if vec is None or len(vec) == 0:
                    raise ValueError(f"Empty vector for doc: {doc.page_content[:50]}...")
                vectors.append(vec.tolist())

    print(f"✅ Generated {len(vectors)} vectors of dimension {len(vectors[0])}")
    return vectors


# -------- Node 4: Upload to Qdrant --------
def upload_to_qdrant(vectors, docs, collection_name="rag_collection", use_openai=True):
    print(f"📡 Uploading vectors to Qdrant collection '{collection_name}'...")
    qdrant_client = get_qdrant_client()
    vector_size = len(vectors[0])

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "page_content": docs[i].page_content,  
                "source": docs[i].metadata.get("source", "")
            }
        )
        for i in range(len(docs))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Uploaded {len(points)} vectors to Qdrant collection '{collection_name}'")
    return f"✅ Qdrant now contains {len(points)} vectors"
