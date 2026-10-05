class QueryExpander:


    def __init__(self):
        self.domain_terms={
            "top up":[
                    "prepaid recharge",
                    "reload",
                    "recharge transactions",
                ],
            "topping up": [
                    "prepaid recharge",
                    "reload",
                    "recharge transactions",
                ],
        }
    
    def expand(self,query):
        expanded_terms=[]
    
        query_lower=query.lower()
    
        for trigger,related_terms in self.domain_terms.items():
            if trigger in query_lower:
                expanded_terms.extend(related_terms)
    
        expanded_query=query + " " + " ".join(expanded_terms)
    
        return expanded_query