from langgraph.graph import StateGraph, START, END
from src.graph.nodes import (
    load_clean_data, 
    chunk_documents, 
    embed_documents, 
    test_similarity_search, 
    upload_to_qdrant
)

def build_pipeline(use_openai=True):
    print("🛠 Building LangGraph pipeline...")

    graph = StateGraph(dict)

    # Nodes
    graph.add_node("load", lambda state: {"data": load_clean_data()})
    graph.add_node("chunk", lambda state: {"docs": chunk_documents(state["data"])})
    
    # Keep docs + vectors
    graph.add_node("embed", lambda state: {
        "vectors": embed_documents(state["docs"], use_openai=use_openai),
        "docs": state["docs"]
    })

    # Test similarity search before Qdrant
    graph.add_node("test_search", lambda state: {
        "vectors": state["vectors"],
        "docs": state["docs"],
        "test_results": test_similarity_search(state["docs"], state["vectors"], use_openai=use_openai)
    })

    graph.add_node("upload", lambda state: {
        "result": upload_to_qdrant(state["vectors"], state["docs"], use_openai=use_openai)
    })

    # Flow
    graph.add_edge(START, "load")
    graph.add_edge("load", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "test_search")
    graph.add_edge("test_search", "upload")
    graph.add_edge("upload", END)

    print("✅ Pipeline built successfully")
    return graph.compile()
