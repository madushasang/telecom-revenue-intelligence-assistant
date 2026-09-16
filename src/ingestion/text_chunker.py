def chunk_by_characters(text:str,chunk_size:int=500,chunk_overlap: int = 100)-> list[str]:
    """
    Split text into fixed-size character chunks with overap.
    """

    if chunk_overlap>=chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks =[]

    step_size=chunk_size - chunk_overlap

    for start in range(0,len(text),step_size):
        end =start + chunk_size
        chunk =text[start:end]

        chunks.append(chunk)

        if end >= len(text):
            break
    return chunks
