from typing import List


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    total = len(text)

    while start < total:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += max(1, chunk_size - overlap)

    return chunks