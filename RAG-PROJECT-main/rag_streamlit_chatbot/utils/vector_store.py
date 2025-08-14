from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


def build_vector_db(documents, persist_directory):
    """Build a Chroma vector database from the given documents."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    except Exception as e:
        raise Exception(f"Error building vector database: {str(e)}")


def save_vector_db(vectorstore):
    """Persist Chroma vector database to disk."""
    try:
        vectorstore.persist()
    except Exception as e:
        raise Exception(f"Error saving vector database: {str(e)}")


def load_vector_db(persist_directory):
    """Load Chroma vector database from disk."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        raise Exception(f"Error loading vector database: {str(e)}")


def get_source_documents(vectorstore):
    """Get unique source document names from the vector database."""
    if not vectorstore:
        return []
    
    try:
        # This is a simplified way to get sources. For a production system,
        # you might store metadata more explicitly.
        all_docs = vectorstore.get()
        sources = [doc.get("metadata", {}).get("source") for doc in all_docs.get("metadatas", [])]
        return sorted(list(set(s for s in sources if s)))
    except Exception as e:
        # Handle cases where the vectorstore might not have a `get` method
        # or the metadata structure is different.
        print(f"Could not retrieve source documents: {e}")
        return []


def add_to_vector_db(vectorstore, documents):
    """Add new documents to an existing Chroma vector database."""
    try:
        vectorstore.add_documents(documents)
    except Exception as e:
        raise Exception(f"Error adding to vector database: {str(e)}")
