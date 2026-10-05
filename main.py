from src.ingestion.document_loader import load_text_document
from src.ingestion.text_chunker import chunk_by_characters
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.markdown_chunker import chunk_by_markdown_sections
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from sentence_transformers import CrossEncoder
from src.retrieval.reranker import Reranker
from src.retrieval.query_expander import QueryExpander
from src.generation.context_builder import build_context
from src.generation.prompt_builder import build_prompt
from src.generation.generator import OllamaGenerator



DOCUMENT_PATH = "data/raw/telecom_revenue_knowledge.md"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main():
    #1.Load document
    document=load_text_document(DOCUMENT_PATH)
    

    #2.Split document into chunks
    #chunks=chunk_by_characters(document,chunk_size=500,chunk_overlap=100)
    chunks=chunk_by_markdown_sections(document,DOCUMENT_PATH)
    
    print(f"Indexed chunks: {len(chunks)}")
    



    query = "another local operator requests the total incoming voice usage from their subscribers to our network in the last week"


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
        {
        "query": "Why subscribers travelling in India using less data?",
        "expected_section": "Roaming Revenue",
    },
    {
        "query": "another local operator requests the total incoming voice usage from their subscribers to our network in the last week",
        "expected_section": "Interconnect Revenue",
    },
    {
        "query": "last week VAS revenues seems too high, Should I consult with the product owner?",
        "expected_section": "Revenue Anomaly Investigation",
    },
    {
        "query": "our OCS voucher CDRs shows highier prepaid revenue in last two days compared to last week",
        "expected_section": "Prepaid Recharge Revenue",
    },
    {
        "query": "total voice minutes usage seems increasing while prepaid voice revenue drops",
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

    retrieval_terms = {
    "Prepaid Recharge Revenue": [
        "Top up",
		"Reload",
		"Prepaid account",
		"OCS subscriber balance",
		"Online recharge",
		"Voucher cards"
    ],

    "Usage Revenue": [
        "voice minutes",
		"chargeable usage",
		"usage charging",
		"traffic volume versus charged amount"
    ],

    "Roaming Revenue": [
        "TAP files",
		"Inbound subscribers",
		"Outbound subscribers",
		"DCH",
		"Roaming Partner",
		"TADIGS"
   
    
    ],

    "Interconnect Revenue": [
        "incoming interconnect traffic"
		"outgoing interconnect traffic"
		"traffic from another operator"
		"traffic to another operator",
		"interconnect settlement",
		"interconnect tariffs",
		"flat rate charging",
		"interconnect billing",
      
    ],

    "Revenue Anomaly Investigation": [
        "unexpected revenue deviation"
		"abnormal revenue trend"
		"revenue trend monitoring"
		"seasonal revenue variation"
		"holiday revenue variation"
    ],
    }
    
   

    query = "Our roaming revenue dropped by 20% yesterday. What caused it?"

    #1.Retrieve
    semantic_retriever=SemanticRetriever(MODEL_NAME)
    semantic_retriever.index(chunks)
    #evaluator=RetrievalEvaluator(semantic_retriever)


    results=semantic_retriever.search(query=query,
    top_k=3,
    min_score=0.0,
    )

    #2.Build context
    context = build_context(results)

    #3.Build Prompt
    prompt = build_prompt(
    question=query,
    context=context,
    )

    #4.Generate

    generator=OllamaGenerator("qwen3:8b")

    answer=generator.generate(prompt)

    print(answer)




    
    
if __name__ == "__main__":
    main()






