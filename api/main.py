from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.llm.rag_pipeline import RAGPipeline

rag = RAGPipeline()

app = FastAPI(
    title="IMF RAG API",
    description="Query the IMF RAG pipeline",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    context: List[str]

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer = rag.query(request.query)
    
    # Retrieve context docs
    context_docs = rag.hybrid_retriever.qdrant_store.similarity_search(request.query, k=3)
    sources = list({doc.metadata.get("source", "unknown") for doc in context_docs})
    context = [doc.page_content for doc in context_docs]
    
    return QueryResponse(answer=answer, sources=sources, context=context)
