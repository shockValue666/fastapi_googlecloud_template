import logging
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader

def pypdf_parser(pdf_url: str, max_retries: int = 3) -> List[Dict]:
    """
    Parse a PDF file using the PyPDFLoader and return its content and metadata.

    Args:
        pdf_url (str): URL of the PDF file to parse.
        max_retries (int): Maximum number of retries in case of failure.

    Returns:
        List[Dict]: List of dictionaries containing page content and metadata for each page.

    Raises:
        Exception: If parsing fails after the maximum number of retries.
    """
    retries = 0
    while retries < max_retries:
        try:
            # Initialize the PyPDFLoader
            loader = PyPDFLoader(file_path=pdf_url, extract_images=True)

            # Lazy load the documents
            docs_lazy = loader.lazy_load()

            # Extract content and metadata
            docs = []
            for doc in docs_lazy:
                docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

            logging.info(f"Successfully parsed {len(docs)} pages from the PDF.")
            return docs

        except Exception as e:
            retries += 1
            logging.error(f"Error parsing PDF. Attempt {retries}/{max_retries}: {e}")
            if retries >= max_retries:
                raise Exception(f"Failed to parse PDF after {max_retries} attempts.") from e

    return []  # Fallback return (shouldn't reach this point)