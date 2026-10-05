from rank_bm25 import BM25Okapi
import re
import numpy as np


class LexicalRetriever:


    def __init__(self):
        self.chunks=[]
        self.bm25=None

    def tokenize(self,text:str) -> list[str]:
        return re.findall(r"\b\w+\b",text.lower())


    def index(self,chunks: list[dict]) -> None:
        self.chunks=chunks

        tokenized_documents=[]

        for chunk in chunks:
            text=chunk["text"]
            tokenized_documents.append(self.tokenize(text))
        self.bm25=BM25Okapi(tokenized_documents)



    def search(self,query:str,top_k:int =3):
        if self.bm25 is None:
            raise ValueError("No documents have been indexed.")

        query_tokens=self.tokenize(query)
        scores=self.bm25.get_scores(query_tokens)
        ranked_indeces=np.argsort(scores)[::-1]

        results=[]

        for index in ranked_indeces[:top_k]:
            results.append((float(scores[index]),self.chunks[index]))

        return results
