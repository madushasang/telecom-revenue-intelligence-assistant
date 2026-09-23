from src.ingestion.document_loader import load_text_document
from src.ingestion.text_chunker import chunk_by_characters
from src.retrieval.semantic_retriever import SemanticRetriever
from src.ingestion.markdown_chunker import chunk_by_markdown_sections
from src.evaluation.retrieval_evaluator import RetrievalEvaluator



DOCUMENT_PATH = "data/raw/telecom_revenue_knowledge.md"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    #1.Load document
    document=load_text_document(DOCUMENT_PATH)
    

    #2.Split document into chunks
    #chunks=chunk_by_characters(document,chunk_size=500,chunk_overlap=100)
    chunks=chunk_by_markdown_sections(document,DOCUMENT_PATH)
    
    print(f"Indexed chunks: {len(chunks)}")

    #3.Create rtriever
    retriever=SemanticRetriever(MODEL_NAME)

    #4.Generate and store document embeddings
    retriever.index(chunks)

    #5.Ask a question
    #query=" The weather was very hot yesterday."

    EVALUATION_QUERIES = [
    {
        "query": "Foreign network charges are not agreeing with what we billed customers.",
        "expected_section": "Roaming Revenue",
    },
    {
        "query": "Traffic from another operator appears to be missing from settlement.",
        "expected_section": "Interconnect Revenue",
    },
    {
        "query": "Yesterday's income looks abnormal. Should I immediately treat it as leakage?",
        "expected_section": "Revenue Anomaly Investigation",
    },
    {
        "query": "Customers are topping up less than usual. What should I investigate?",
        "expected_section": "Prepaid Recharge Revenue",
    },
    {
        "query": "Call volumes increased but the money collected did not increase.",
        "expected_section": "Usage Revenue",
    },
]

    NEGATIVE_QUERIES  = [
    {
        "query": "What will the weather be tomorrow?",
        "expected_section": None,
    },
    {
        "query": "How do I replace a car battery?",
        "expected_section": None,
    },
    {
        "query": "What is the best way to learn Python?",
        "expected_section": None,
    },
    {
        "query": "How can I improve employee motivation?",
        "expected_section": None,
    },
    {
        "query": "Why did the stock market fall yesterday?",
        "expected_section": None,
    },
]
    thresholds = [
        0.00,
        0.10,
        0.20,
        0.25,
        0.30,
        0.35,
    ]
    evaluator=RetrievalEvaluator(retriever)

    for threshold in thresholds:

        metrics=evaluator.evaluate_positive(EVALUATION_QUERIES,top_k=3,min_score=threshold)
        #print(metrics)

        negative_metrics=evaluator.evaluate_negatives(NEGATIVE_QUERIES,top_k=1,min_score=threshold)
        #print(negative_metrics)

        print(f"Threshold={threshold} | Hit@1={metrics["hit_rate_at_1"]:.2%} | Hit@k={metrics["hit_rate_at_k"]:.2%} | Rejection= {negative_metrics["rejection_rate"]:.2%}")


    

if __name__ == "__main__":
    main()






