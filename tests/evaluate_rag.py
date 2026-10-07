
from src.rag.rag_service import RAGService
from src.retrieval.semantic_retriever import SemanticRetriever
from src.ingestion.markdown_chunker import chunk_by_markdown_sections
from src.ingestion.document_loader import load_text_document
from src.generation.generator import OllamaGenerator
from src.evaluation.rag_evaluator import RAGEvaluator
from src.config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    RETRIEVAL_TOP_K,
)




DOCUMENT_PATH = "data/raw/telecom_revenue_knowledge.md"


test_cases = [
    {
        "query": "Why might TAP records not match roaming billing?",
        "expected_behavior": "answerable",
        "expected_section": "Roaming Revenue",
        "expected_concepts": [
            "missing TAP records",
            "delayed TAP files",
            "incorrect roaming tariffs",
            "currency conversion",
            "rejected records",
            "duplicate records",
            "incorrect mapping",
        ],
        "min_required_concepts": 2,
    },
    {
        "query": "What was our total prepaid recharge revenue yesterday?",
        "expected_behavior": "insufficient_context",
    },
	]

document=load_text_document(DOCUMENT_PATH)
chunks=chunk_by_markdown_sections(document,DOCUMENT_PATH)

semantic_retriever=SemanticRetriever(model_name=EMBEDDING_MODEL)
semantic_retriever.index(chunks)

generator=OllamaGenerator(GENERATION_MODEL)
rag_evaluator=RAGEvaluator()


rag_service=RAGService(retriever=semantic_retriever,generator=generator)

passed_cases = 0
for test_case in test_cases:
    result = rag_service.ask(test_case["query"],top_k=RETRIEVAL_TOP_K)

    evaluation = rag_evaluator.evaluate(
        test_case,
        result["answer"],
    )

    if evaluation["passed"]:
        passed_cases += 1

    print(f"\nQuery: {test_case['query']}")
    print(f"Evaluation: {evaluation}")

total_cases = len(test_cases)
pass_rate = passed_cases / total_cases

print("\nFINAL RAG EVALUATION")
print(f"Passed: {passed_cases}/{total_cases}")
print(f"Pass rate: {pass_rate:.2%}")