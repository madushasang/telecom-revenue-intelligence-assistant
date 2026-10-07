class RAGEvaluator:

    def __init__(self):
        self.insufficient_phrases =[ "insufficient",
            "not enough information",
            "not provided",
            "does not contain",
            "cannot determine",
            ]

    def evaluate(self,test_case,answer):

        expected_behavior=test_case["expected_behavior"]

        if expected_behavior=="answerable":
            detected_concepts=0
            expected_concepts=test_case["expected_concepts"]

            for expected_concept in expected_concepts:
                if expected_concept.lower() in answer.lower():
                                
                    detected_concepts += 1
            concept_pass = (detected_concepts >= test_case["min_required_concepts"])
                
            source_pass=(test_case["expected_section"].lower() in answer.lower())
                
            case_pass = concept_pass and source_pass
            return {
                    "passed": case_pass,
                    "concept_pass": concept_pass,
                    "source_pass" : source_pass,
                    "detected_concepts": detected_concepts,
                        }

            

        elif expected_behavior=="insufficient_context":
            insufficient_pass = False
            
            for phrase in self.insufficient_phrases:
            
                if phrase.lower() in answer.lower():
                    insufficient_pass = True
                    break
            return {
                "passed": insufficient_pass,
                "insufficient_context_pass": insufficient_pass,
            }