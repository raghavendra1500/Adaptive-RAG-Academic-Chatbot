"""
Prompt Builder
---------------
Creates a grounded prompt for the LLM using retrieved document chunks.
"""


class PromptBuilder:

    def __init__(self):
        pass

    def build_prompt(self, query, retrieved_chunks):
        """
        Builds a structured prompt for the LLM.
        """

        if not retrieved_chunks:
            context = "No relevant document context found."
        else:
            context = ""

            for rank, chunk in enumerate(retrieved_chunks, start=1):

                context += f"""
====================================================
Document Chunk {rank}

Rank : {rank}
Page Number : {chunk['page_number']}
Similarity : {chunk['score']:.4f}

Document Text:
{chunk['text']}

====================================================

"""

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