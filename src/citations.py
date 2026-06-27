"""
citations.py
------------
Extracts and formats citations from retrieved document chunks.
"""


class CitationGenerator:
    """
    Generates citations from retrieved chunks.
    """

    def __init__(self):
        pass

    def get_page_numbers(self, chunks):
        """
        Returns a sorted list of unique page numbers.
        """

        pages = sorted(
            list(
                {
                    chunk["page_number"]
                    for chunk in chunks
                    if "page_number" in chunk
                }
            )
        )

        return pages

    def format_citations(self, chunks):
        """
        Returns citations in the format:
        Page 10, Page 12, Page 18
        """

        pages = self.get_page_numbers(chunks)

        if not pages:
            return "No citations available."

        return ", ".join([f"Page {page}" for page in pages])

    def generate(self, chunks):
        """
        Returns a citation dictionary.
        """

        pages = self.get_page_numbers(chunks)

        return {
            "pages": pages,
            "citation_text": self.format_citations(chunks),
            "total_pages": len(pages)
        }