"""
Prompt Builder
---------------
Creates a grounded prompt for the LLM using retrieved document chunks.
"""


class PromptBuilder:

    def __init__(self):
        self.task_instructions = {
            "definition": "Define the concept precisely, then add only document-supported details.",
            "comparison": "Compare the requested items in a compact academic structure and cite pages for each point.",
            "methodology": "Explain the method or procedure in ordered steps when the context supports it.",
            "summary": "Summarize the most important document-supported ideas without adding outside context.",
            "factual": "Answer directly and briefly, using exact document evidence where possible.",
            "explanation": "Explain the reasoning using only evidence from the retrieved context.",
            "open": "Answer in a concise academic style using only the retrieved context.",
        }

    def build_prompt(self, query, retrieved_chunks, query_type="open", conversation_history=None):
        """
        Builds a structured prompt for the LLM.
        """

        if not retrieved_chunks:
            context = "No relevant document context found."
        else:
            context = ""

            for rank, chunk in enumerate(retrieved_chunks, start=1):

                source = chunk.get("source_file", "Uploaded PDF")
                section = chunk.get("section_title") or "Unknown section"
                context += f"""
====================================================
Document Chunk {rank}

Rank : {rank}
Source : {source}
Page Number : {chunk['page_number']}
Section : {section}
Similarity : {chunk['score']:.4f}

Document Text:
{chunk['text']}

====================================================

"""

        history_text = self._format_history(conversation_history or [])
        task_instruction = self.task_instructions.get(query_type, self.task_instructions["open"])

        prompt = f"""
You are an Academic Document Assistant.

Your job is to answer questions ONLY using the supplied document context.

Instructions:

1. Use ONLY the provided document context.
2. Never use your own knowledge.
3. Never guess or hallucinate information.
4. If the answer cannot be found in the document, reply exactly:

The uploaded academic document does not contain enough information to answer this question.

5. Combine information from multiple chunks when appropriate.
6. Mention supporting page numbers.
7. Keep the answer concise and academic.
8. Task-specific instruction: {task_instruction}

====================================================
CONVERSATION MEMORY
====================================================

{history_text}

====================================================
QUESTION
====================================================

{query}

====================================================
DOCUMENT CONTEXT
====================================================

{context}

====================================================
REQUIRED OUTPUT FORMAT
====================================================

Answer:
<Your answer>

Supporting Pages:
<Page numbers>

Confidence:
High / Medium / Low
"""

        return prompt

    def _format_history(self, conversation_history):
        if not conversation_history:
            return "No prior conversation."

        recent_items = conversation_history[-3:]
        lines = []

        for item in recent_items:
            question = item.get("question", "")
            answer = item.get("answer", "")
            lines.append(f"Previous Question: {question}\nPrevious Answer: {answer[:500]}")

        return "\n\n".join(lines)
