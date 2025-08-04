# src/retrieval/reranker.py
from sentence_transformers import CrossEncoder
import re

class Reranker:
    def rerank(self, query, docs, top_k=5):
        # Simple numeric prioritization
        scored_docs = []
        for doc in docs:
            score = doc.metadata.get("score", 0)
            # Boost if contains numbers or keywords
            if re.search(r"\d{4}|\d+\.\d+|Annex|Table", doc.page_content):
                score += 2
            scored_docs.append((doc, score))
        # Sort by score
        ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
