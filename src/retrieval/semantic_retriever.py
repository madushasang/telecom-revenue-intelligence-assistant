import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self,model_name:str):
        self.model=SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def index(self,chunks:list[dict]) -> None:
        """
        Convert document chunks into embeddings and store them.
        """

        self.chunks=chunks

        texts=[chunk["text"] for chunk in chunks]

        self.embeddings=self.model.encode(
            texts,convert_to_numpy=True,
        )

    def search(
            self,
            query:str,
            top_k:int =3,
            min_score: float = 0.0,       
    )->list[tuple[float,str]]:
        """
        Find the chunks most semantically similar to the query.
        """

        if self.embeddings is None:
            raise ValueError("No documents have been indexed.")

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
        )

        query_norm=np.linalg.norm(query_embedding)

        document_norms=np.linalg.norm(
            self.embeddings,
            axis=1,
        )

        dot_products=self.embeddings @ query_embedding
        similarities=dot_products/(document_norms*query_norm)

        ranked_indices = np.argsort(similarities)[::-1]

        results =[]

        for index in ranked_indices[:top_k]:
            if similarities[index]>=min_score:
                results.append(
                    (
                        float(similarities[index]),
                        self.chunks[index],
                    )
                )
        return results

