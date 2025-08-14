from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_split(pdf_path):
    """Load and split PDF documents into chunks."""
    try:
        # Verify file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No content found in the PDF file")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = splitter.split_documents(documents)

        if not split_docs:
            raise ValueError("Failed to split documents into chunks")

        return split_docs

    except Exception as e:
        raise Exception(f"Error loading and splitting PDF: {str(e)}")
