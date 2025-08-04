from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

class RAGPipeline:
    def __init__(self):
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load Qdrant collection
        print("📡 Loading documents from Qdrant...")
        self.qdrant_store = Qdrant.from_existing_collection(
            collection_name="rag_collection",
            embedding=embeddings,
            location="localhost",
            port=6333
        )
        print("✅ Loaded documents from Qdrant.")

        # Hybrid retriever (BM25 disabled for now)
        self.hybrid_retriever = HybridRetriever(self.qdrant_store)
        self.reranker = Reranker()
        self.llm = ollama

    def query(self, question):
        # 1️⃣ Retrieve
        candidates = self.hybrid_retriever.search(question, k=15)
        ranked = self.reranker.rerank(question, candidates, top_k=5)

        # 2️⃣ Debug context
        context = "\n".join([doc.page_content for doc in ranked])
        print("\n🔍 [DEBUG] Final Context Sent to LLM:\n")
        for i, doc in enumerate(ranked, 1):
            print(f"--- Doc {i} ---\n{doc.page_content}\n")

        # 3️⃣ Prompt
        prompt = f"""
        You are an expert summarizing IMF reports.
        Use ONLY the context to extract:
        - Specific numbers (percentages, GDP growth, inflation rates, member counts)
        - Specific trends (increase/decrease, regional comparisons)

        Context:
        {context}

        Question: {question}
        """
        response = self.llm.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
