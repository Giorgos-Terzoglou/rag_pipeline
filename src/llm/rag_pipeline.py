from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.reranker import Reranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
import ollama

class RAGPipeline:
    def __init__(self):
        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Qdrant store
        self.qdrant_store = Qdrant.from_existing_collection(
            embedding=embeddings,
            collection_name="rag_collection",
            url="http://localhost:6333"
        )

        # BM25 retriever from Qdrant payload
        client = QdrantClient(host="localhost", port=6333)
        bm25_docs = []
        offset = None
        batch_size = 100
        batch_count = 0

        print("📡 Loading BM25 documents from Qdrant in batches...")
        while True:
            scroll_res, next_offset = client.scroll(
                collection_name="rag_collection",
                limit=batch_size,
                offset=offset
            )
            
            if not scroll_res:  # No more results
                break
            
            for point in scroll_res:
                if "text" in point.payload and point.payload["text"].strip():
                    bm25_docs.append(point.payload["text"])
                elif "page_content" in point.payload and point.payload["page_content"].strip():
                    bm25_docs.append(point.payload["page_content"])
                        
            batch_count += 1
            print(f"  ✅ Loaded batch {batch_count} ({len(bm25_docs)} docs total)")
            
            if next_offset is None:  # End of collection
                break
            offset = next_offset

        print(f"📡 Finished loading BM25 docs. Total: {len(bm25_docs)}")

        self.bm25_retriever = BM25Retriever(bm25_docs, collection_name="rag_collection")

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(self.qdrant_store, self.bm25_retriever)

        # Reranker
        self.reranker = Reranker()

    def query(self, question):
        # 1️⃣ Retrieve candidates
        candidates = self.hybrid_retriever.search(question, k=15)

        # 2️⃣ Rerank
        ranked = self.reranker.rerank(question, candidates, top_k=5)

        # 3️⃣ Debug context
        context = "\n".join([doc.page_content for doc in ranked])
        print("\n🔍 [DEBUG] Final Context Sent to LLM:\n")
        for i, doc in enumerate(ranked, 1):
            print(f"--- Doc {i} ---\n{doc.page_content}\n")

        # 4️⃣ Prompt
        prompt = f"""
        You are an expert in summarizing IMF reports.
        Use ONLY the following context to extract:
        - Specific numbers (percentages, GDP growth, inflation rates, member counts)
        - Specific trends (increase/decrease over time, comparisons between regions)

        Context:
        {context}

        Question: {question}
        """
        
        # 5️⃣ Call Llama 3.1 locally
        response = ollama.chat(model="llama3.1:8b", messages=[{"role":"user", "content":prompt}])
        return response["message"]["content"]
