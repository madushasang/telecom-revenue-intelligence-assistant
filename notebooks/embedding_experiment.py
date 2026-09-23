from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def cosine_similarity(vector_a,vector_b):
    dot_product = np.dot(vector_a,vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    similarity = dot_product/(magnitude_a * magnitude_b)

    return similarity

def main():
    model=SentenceTransformer(MODEL_NAME)
    sentences = [
        "Prepaid recharge revenue decreased.",
        "Recharge income dropped.",
        "Roaming billing records are missing.",
        "The weather is sunny today.",
        "Top-up value declined significantly.",
    ]


    embeddings =model.encode(sentences)

    reference = embeddings[0]

    print(f"Number of sentences: {len(sentences)}")
    print(f"embedding shape: {embeddings.shape}")

    print()

    for sentence, embedding in zip(sentences, embeddings):
        print("="*60)
        print(f"Sentence :{sentence}")
        print(f"Vector dimensions: {len(embedding)}")
        print(f"first 10 values: {embedding[:10]}")

    print("\nCOSINE SIMILARITY")
    print("=" * 60)

    for sentence,embedding in zip(sentences,embeddings):
        score = cosine_similarity(reference,embedding)

        print(f"{score:.4f} |"
              f"{sentence}"
              )

if __name__=="__main__":
    main()