import re


def prosody_split(text: str, max_chars: int = 150) -> list[str]:
    """
    Split text on natural prosody boundaries.
    Priority:
      1) Strong punctuation (.?!)
    """
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip() if current else word

        current = candidate

        # Strong punctuation: split immediately.
        if re.search(r"[.?!]$", word):
            chunks.append(current.strip())
            current = ""
            continue

    if current:
        chunks.append(current.strip())

    return chunks
