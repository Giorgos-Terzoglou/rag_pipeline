from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

class BM25Retriever:
    def __init__(self, collection_name="rag_collection"):
        print("📡 Loading BM25 documents from Qdrant (safe scroll)...")
        self.client = QdrantClient(host="localhost", port=6333)

        all_docs = []
        next_offset = None
        batch_size = 50  # keep small to avoid long URL issues

        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=True
            )

            if not points:
                break

            for p in points:
                text = p.payload.get("text", "")
                if isinstance(text, str) and len(text.strip()) > 20:
                    all_docs.append(text.strip())

            if next_offset is None:  # ✅ stop when no more pages
                break

        if not all_docs:
            print("⚠️ No valid BM25 documents found.")
            self.bm25 = None
        else:
            print(f"✅ Loaded {len(all_docs)} clean BM25 documents for BM25.")
            tokenized_corpus = [doc.split() for doc in all_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.all_docs = all_docs

    def search(self, query, k=10):
        if self.bm25 is None:
            print("⚠️ BM25 is not initialized.")
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.all_docs[i] for i in top_k_idx]
