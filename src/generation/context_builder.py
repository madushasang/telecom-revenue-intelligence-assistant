def build_context(results):
    context_parts=[]

    for score,chunk in results:
        
        text=chunk["text"]

        context_parts.append(text)
    context="\n\n".join(context_parts)

    return context