from pathlib import Path

def load_text_document(file_path: str) -> str:
    """
    Load a UTF-8 text document and return its contents as a string.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found:{file_path}")

    return path.read_text(encoding="utf-8")
