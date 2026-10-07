from src.ingestion.document_loader import load_text_document
from src.ingestion.markdown_chunker import chunk_by_markdown_sections
from src.retrieval.semantic_retriever import SemanticRetriever
from src.generation.generator import OllamaGenerator
from src.rag.rag_service import RAGService
from src.config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    RETRIEVAL_TOP_K,
)



DOCUMENT_PATH = "data/raw/telecom_revenue_knowledge.md"





def main():
    #1.Load document
    document=load_text_document(DOCUMENT_PATH)
    

    #2.Split document into chunks
    chunks=chunk_by_markdown_sections(document,DOCUMENT_PATH)
    



  
    # Build retriever
    semantic_retriever=SemanticRetriever(EMBEDDING_MODEL)
    semantic_retriever.index(chunks)

    # Build generator
    generator=OllamaGenerator(GENERATION_MODEL)
    

    # Build application service
    rag_service=RAGService(retriever=semantic_retriever,generator=generator)


    # Ask the assistant
    question = "Why might TAP records not match roaming billing?"

    result = rag_service.ask(question,top_k=RETRIEVAL_TOP_K)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"Rank {source['rank']} | "
            f"{source['section']} | "
            f"{source['score']:.4f}"
        )
    
    print("\nTIMING:")
    print(
        f"Retrieval: "
        f"{result['timing']['retrieval_ms']:.2f} ms"
    )
    print(
        f"Generation: "
        f"{result['timing']['generation_ms'] / 1000:.2f} sec"
    )
    
    
if __name__ == "__main__":
    main()






