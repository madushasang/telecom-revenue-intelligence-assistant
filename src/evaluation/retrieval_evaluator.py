
class RetrievalEvaluator:

    def __init__(self,retriever):
        self.retriever=retriever


    def evaluate_positive(self,test_cases,top_k,min_score):
        hits_at_1=0
        hits_at_k=0

        if not test_cases:
            return("No test cases provided")


        for test_case in test_cases:
            query=test_case["query"]
            expected=test_case["expected_section"]
            results=self.retriever.search(query,top_k,min_score)
            if not results:
                continue
            else:
                results_sections=[result[1]["section"] for result in results]
                if expected==results_sections[0]:
                    hits_at_1+=1
                if expected in results_sections:
                    hits_at_k+=1

        hit_rate_at_1=hits_at_1/len(test_cases)
        hit_rate_at_k=hits_at_k/len(test_cases)
        return{"hit_rate_at_1" : hit_rate_at_1, "hit_rate_at_k" : hit_rate_at_k}

    def evaluate_negatives(self,test_cases,top_k,min_score):
        correct_rejections = 0

        if not test_cases:
            return("No test cases provided")

        for test_case in test_cases:
            query = test_case["query"]

            results=self.retriever.search(query,top_k,min_score)
            if not results:
                correct_rejections+=1

        rejection_rate=correct_rejections/len(test_cases)
        return{"rejection_rate":rejection_rate}


    def evaluate_positive_reranked(self,test_cases,reranker,top_k,min_score):
        hits_at_1=0
        hits_at_k=0

        if not test_cases:
                    return("No test cases provided")

        for test_case in test_cases:
            query=test_case["query"]
            expected=test_case["expected_section"]

            results=self.retriever.search(query,top_k,min_score)

            if not results:
                continue
            reranked_results=reranker.rerank(query,results)

            retrieved_sections=[result[2]["section"] for result in reranked_results]

            if expected == retrieved_sections[0]:
                hits_at_1 +=1

            if expected in retrieved_sections:
                hits_at_k +=1

            #if expected != retrieved_sections[0]:
#
            #    print("\nRERANK MISS")
            #    print("Query    :", query)
            #    print("Expected :", expected)
#
            #    for result in reranked_results:
            #        rerank_score = result[0]
            #        retrieval_score = result[1]
            #        section = result[2]["section"]
#
            #        print(
            #            f"{section:35} "
            #            f"retrieval={retrieval_score:.4f} "
            #            f"rerank={rerank_score:.4f}"
            #        )

        hit_rate_at_1=hits_at_1/len(test_cases)
        hit_rate_at_k=hits_at_k/len(test_cases)
        return{"hit_rate_at_1" : hit_rate_at_1, "hit_rate_at_k" : hit_rate_at_k}

    def evaluate_positive_hybrid(self,test_cases,hybrid_retriever,top_k):
            hits_at_1=0
            hits_at_k=0
    
            if not test_cases:
                        return("No test cases provided")
    
            for test_case in test_cases:
                query=test_case["query"]
                expected=test_case["expected_section"]
    
                results=hybrid_retriever.search(query,top_k=top_k,candidate_k=5)

                if not results:
                     continue
                retrieved_sections=[result[0] for result in results]

                if retrieved_sections[0]==expected:
                     hits_at_1+=1
                if expected in retrieved_sections:
                     hits_at_k+=1

            hit_rate_at_1=hits_at_1/len(test_cases)
            hit_rate_at_k=hits_at_k/len(test_cases)
            return{"hit_rate_at_1" : hit_rate_at_1, "hit_rate_at_k" : hit_rate_at_k}