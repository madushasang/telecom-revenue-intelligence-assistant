from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self,model_name):
        self.model=CrossEncoder(model_name)


    def rerank(self,query,results):

        if not results:
            return[]

        pairs=[[query,result[1]["text"]] for result in results]

        scores=self.model.predict(pairs)

        reranked_results=[]

        for i in range(len(results)):
            reranked_results.append(
                (
                    float(scores[i]),
                    #results[i][0],
                    results[i][1],
                )
            )

        reranked_results=sorted(reranked_results,key=lambda x:x[0],reverse=True)
        return reranked_results