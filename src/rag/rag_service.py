import time

from src.generation.context_builder import build_context
from src.generation.prompt_builder import build_prompt

class RAGService:

    def __init__(self,retriever,generator):
        self.retriever=retriever
        self.generator=generator

    def ask(self,question:str,top_k:int=3):
        #1.Retrieve
        
        retrieval_start=time.perf_counter()
        
        results=self.retriever.search(query=question,top_k=top_k)
        
        retrieval_end=time.perf_counter()
        
        retrieval_time_ms=(retrieval_end-retrieval_start) *1000


        #2.Build context
        
        context = build_context(results)

        #3.Build Prompt

        prompt = build_prompt(question=question,context=context)

        #print(f"PROMPT: \n {prompt}")

        #4.Generate

        generation_start=time.perf_counter()
        answer=self.generator.generate(prompt)
        generation_end=time.perf_counter()

        generation_time_ms=(generation_end - generation_start) *1000

        # Prepare source metadata

        sources=[]
        for rank,(score,chunk) in enumerate(results,start=1):
            sources.append({"rank":rank,"section":chunk["section"],"score":float(score)})

        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "timing": {
                "retrieval_ms": retrieval_time_ms,
                "generation_ms": generation_time_ms,
            },
        }