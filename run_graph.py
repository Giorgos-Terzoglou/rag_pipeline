import argparse
from src.graph.build_graph import build_pipeline
from src.llm.rag_pipeline import RAGPipeline

parser = argparse.ArgumentParser(description="LangGraph RAG Pipeline")
parser.add_argument("--build", action="store_true", help="Rebuild Qdrant index from raw data")
parser.add_argument("--query", type=str, help="Query the RAG pipeline")
args = parser.parse_args()

if args.build:
    print("🚀 Starting LangGraph RAG Build...")
    graph = build_pipeline(use_openai=False)
    result = graph.invoke({})
    print("🏁 Build finished:", result)

elif args.query:
    print("🚀 Starting LangGraph RAG Query...")
    rag = RAGPipeline()
    answer = rag.query(args.query)
    print("\n🤖 LLM Answer:\n", answer)

else:
    parser.print_help()
