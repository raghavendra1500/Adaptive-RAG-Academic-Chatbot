"""
PDF Processing Module
---------------------
Extracts text from academic PDFs while preserving page numbers.
"""

import fitz  # PyMuPDF
import os


class PDFProcessor:

    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Clean extracted text.
        """

        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        while "  " in text:
            text = text.replace("  ", " ")

        return text.strip()

    def extract_text(self, pdf_path):
        """
        Extract text page-wise.

        Returns:
            List of dictionaries
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"{pdf_path} not found.")

        document = fitz.open(pdf_path)

        pages = []

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            text = page.get_text()

            text = self.clean_text(text)

            if len(text) == 0:
                continue

            pages.append(
                {
                    "page_number": page_number + 1,
                    "text": text
                }
            )

        document.close()

        return pages