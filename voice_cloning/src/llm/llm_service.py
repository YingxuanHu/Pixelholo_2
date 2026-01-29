import os
from typing import Generator, List, Dict

from dotenv import load_dotenv
from groq import Groq


class LLMService:
    def __init__(self, system_prompt: str = "You are a helpful, concise AI assistant."):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️ WARNING: GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def stream_response(
        self,
        user_input: str,
        min_words: int = 8,
        min_chars: int = 20,
    ) -> Generator[str, None, None]:
        """
        Sends text to Groq (Llama 3) and yields COMPLETE SENTENCES as they are generated.
        """
        self.history.append({"role": "user", "content": user_input})

        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=self.history,
                stream=True,
                temperature=0.7,
            )
        except Exception as e:
            yield f"Error calling Groq: {str(e)}"
            return

        buffer = ""
        full_response_text = ""

        print("🧠 Llama 3 Thinking...", end="", flush=True)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                buffer += token
                full_response_text += token

                if any(punct in token for punct in [".", "?", "!", "\n"]):
                    if buffer.strip().endswith("Mr.") or buffer.strip().endswith("Dr."):
                        continue

                    sent = buffer.strip()
                    if sent:
                        word_count = len(sent.split())
                        if word_count < min_words or len(sent) < min_chars:
                            continue
                        yield sent
                        buffer = ""

        if buffer.strip():
            yield buffer.strip()

        self.history.append({"role": "assistant", "content": full_response_text})
        print(" Done.")
