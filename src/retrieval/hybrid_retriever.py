class HybridRetriever:
    def __init__(self,semantic_retriever,lexical_retriever):
        self.semantic_retriever=semantic_retriever
        self.lexical_retriever=lexical_retriever


    def search(self,query:str,top_k:int=3,candidate_k:int=5,rrf_k:int=60):
        semantic_results=self.semantic_retriever.search(query,top_k=candidate_k,min_score=0.0)

        lexical_results=self.lexical_retriever.search(query,top_k=candidate_k)

        rrf_scores={}
        

        for rank,result in enumerate(semantic_results,start=1):
            chunk=result[1]
            section=chunk["section"]

            rrf_value=1/(rrf_k+rank)

            if section not in rrf_scores:
                rrf_scores[section] = {
                    "score": 0.0,
                    "chunk": chunk,
                }

            rrf_scores[section]["score"] += rrf_value

        for rank,result in enumerate(lexical_results,start=1):
            chunk=result[1]
            section=chunk["section"]

            rrf_value=1/(rrf_k+rank)

            if section not in rrf_scores:
                            rrf_scores[section] = {
                                "score": 0.0,
                                "chunk": chunk,
                            }
            
            rrf_scores[section]["score"] += rrf_value

        results = []

        for data in rrf_scores.values():
            results.append(
                (data["score"], data["chunk"])
            )

        results = sorted(
            results,
            key=lambda x: x[0],
            reverse=True,
        )

        return results[:top_k]

