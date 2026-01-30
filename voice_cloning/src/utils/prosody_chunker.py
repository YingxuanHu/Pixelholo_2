import re


def prosody_split(text: str, max_chars: int = 150) -> list[str]:
    """
    Split text on natural prosody boundaries.
    Priority:
      1) Strong punctuation (.?!)
      2) Weak punctuation (,;:) only if chunk is getting long
      3) Hard length limit at a word boundary
    """
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip() if current else word

        # Emergency split if adding this word would exceed max_chars.
        if len(candidate) > max_chars and current:
            # Try to split at the last weak punctuation inside current.
            match = re.search(r"^(.*?)([,;:])\s+(.*)$", current)
            if match:
                good_part = (match.group(1) + match.group(2)).strip()
                remainder = match.group(3).strip()
                if good_part:
                    chunks.append(good_part)
                    current = f"{remainder} {word}".strip()
                    continue
            # Hard split at current.
            chunks.append(current.strip())
            current = word
            continue

        current = candidate

        # Strong punctuation: split immediately.
        if re.search(r"[.?!]$", word):
            chunks.append(current.strip())
            current = ""
            continue

        # Weak punctuation: split only if already long enough.
        if re.search(r"[,;:]$", word) and len(current) > 60:
            chunks.append(current.strip())
            current = ""

    if current:
        chunks.append(current.strip())

    return chunks
