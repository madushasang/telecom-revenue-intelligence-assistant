def chunk_by_markdown_sections(text:str,source:str)->list[dict]:
    """
    Split Markdown text using level-2 headings as semantic boundaries.
    """

    parts=text.split("\n##")

    chunks=[]

    for part in parts[1:]:
        section=part.split("\n")[0].strip()
        chunk_txt=f"##{part}"
        chunk={
            "text":chunk_txt,
            "section":section,
            "source":source
        }

        chunks.append(chunk)



    return chunks