
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
        
