from src.ingestion.document_loader import load_text_document
from src.ingestion.text_chunker import chunk_by_characters

DOCUMENT_PATH="data/raw/telecom_revenue_knowledge.md"

def main():
    document = load_text_document(DOCUMENT_PATH)

    chunks=chunk_by_characters(document,chunk_size=500,chunk_overlap=100)

    print(f"Document characters:{len(document)}")
    print(f"Number of chucnks: {len(chunks)}")

    for index,chunk in enumerate(chunks,start=1):
        print("\n" + "=" *60)
        print(f"CHUNK {index} | Characters:{len(chunk)}")

        print("=" * 60)

        print(chunk)

if __name__ == "__main__":
    main()