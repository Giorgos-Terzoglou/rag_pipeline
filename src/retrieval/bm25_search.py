from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from langchain.schema import Document 

class BM25Retriever:
    def __init__(self, docs, collection_name="rag_collection"):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        
        # If no docs were passed, load them from Qdrant
        if not docs:
            self.docs = []
            offset = None
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset
                )
                if not points:
                    break
                for point in points:
                    if "text" in point.payload and point.payload["text"].strip():
                        self.docs.append(point.payload["text"])
                    elif "page_content" in point.payload and point.payload["page_content"].strip():
                        self.docs.append(point.payload["page_content"])
                if next_offset is None:
                    break
                offset = next_offset
        else:
            self.docs = [d for d in docs if d.strip()]

        # Initialize BM25
        if self.docs:
            tokenized_corpus = [doc.split() for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print("⚠️ Warning: BM25 corpus is empty.")
            self.bm25 = None

    def search(self, query, k=10):
        if self.bm25 is None:
            print("⚠️ BM25 is not initialized.")
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        # Return Document + score pairs
        return [(Document(page_content=self.docs[i], metadata={}), scores[i]) for i in top_k_idx]
