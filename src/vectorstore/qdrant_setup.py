from qdrant_client import QdrantClient

def get_qdrant_client(host="localhost", port=6333):
    return QdrantClient(host=host, port=port)
