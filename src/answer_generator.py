"""
Answer Generator
----------------
Generates grounded answers using Ollama.
"""

import time
import ollama
from ollama import ResponseError

from config import OLLAMA_MODEL


class AnswerGenerator:

    def __init__(self):
        self.model = OLLAMA_MODEL

    def generate_answer(self, prompt):
        """
        Sends prompt to Ollama and returns
        answer, response time and model name.
        """

        start_time = time.time()

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        except ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama and make sure llama3.2:3b is available."
            ) from exc
        except ResponseError as exc:
            raise RuntimeError(
                f"Ollama returned an error for model {self.model}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Answer generation failed. Check that Ollama is running locally."
            ) from exc

        end_time = time.time()

        answer = response["message"]["content"]

        return {
            "answer": answer,
            "model": self.model,
            "response_time": round(end_time - start_time, 2)
        }
