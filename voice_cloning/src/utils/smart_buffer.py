import re


class SmartStreamBuffer:
    def __init__(self, min_chunk_size: int = 40, max_chunk_size: int = 150) -> None:
        self.buffer = ""
        self.min_chunk = max(1, min_chunk_size)
        self.max_chunk = max(self.min_chunk + 1, max_chunk_size)

    def add_token(self, token: str) -> str | None:
        self.buffer += token
        return self._try_flush()

    def _try_flush(self, force: bool = False) -> str | None:
        text = self.buffer
        length = len(text)

        if not force and length < self.min_chunk:
            return None

        match_strong = re.search(r"([.?!])(\s+|$)", text)
        if match_strong:
            split_idx = match_strong.end()
            chunk = text[:split_idx].strip()
            self.buffer = text[split_idx:].lstrip()
            return chunk

        if length > 80:
            match_weak = re.search(r"([,;:])(\s+)", text)
            if match_weak:
                split_idx = match_weak.end()
                chunk = text[:split_idx].strip()
                self.buffer = text[split_idx:].lstrip()
                return chunk

        if length >= self.max_chunk:
            last_space = text.rfind(" ")
            if last_space != -1:
                chunk = text[:last_space].strip()
                self.buffer = text[last_space:].lstrip()
                return chunk

        if force and text.strip():
            self.buffer = ""
            return text.strip()

        return None

    def flush(self) -> str | None:
        return self._try_flush(force=True)
