"""
Adaptive Chunking Module
------------------------
Creates intelligent chunks for academic documents.
"""

import re

from config import (
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP
)


class AdaptiveChunker:

    def __init__(
        self,
        min_size=MIN_CHUNK_SIZE,
        max_size=MAX_CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    ):

        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap

    def split_into_sentences(self, text):
        """
        Split paragraph into sentences.
        """

        sentences = re.split(r'(?<=[.!?])\s+', text)

        return [
            s.strip()
            for s in sentences
            if s.strip()
        ]

    def adaptive_chunking(self, pages):

        chunks = []

        chunk_id = 1

        for page in pages:

            page_number = page["page_number"]

            sentences = self.split_into_sentences(
                page["text"]
            )

            current_chunk = ""

            for sentence in sentences:

                # Sentence fits

                if len(current_chunk) + len(sentence) <= self.max_size:

                    current_chunk += " " + sentence

                else:

                    chunks.append({

                        "chunk_id": chunk_id,

                        "page_number": page_number,

                        "text": current_chunk.strip(),

                        "length": len(current_chunk)

                    })

                    chunk_id += 1

                    # overlap

                    overlap_text = current_chunk[-self.overlap:]

                    current_chunk = overlap_text + " " + sentence

            if current_chunk:

                chunks.append({

                    "chunk_id": chunk_id,

                    "page_number": page_number,

                    "text": current_chunk.strip(),

                    "length": len(current_chunk)

                })

                chunk_id += 1

        return chunks

    def chunk_pages(self, pages):
        """
        Public wrapper used by the pipeline.
        """

        return self.adaptive_chunking(pages)
