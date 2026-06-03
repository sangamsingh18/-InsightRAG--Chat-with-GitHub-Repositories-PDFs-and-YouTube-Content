import os
import re
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings_model

# Setup logging
logger = logging.getLogger(__name__)

DB_DIR = "chroma_db"

def sanitize_collection_name(name: str) -> str:
    """
    Sanitizes a repository name to comply with ChromaDB collection naming rules:
    1. Length between 3 and 63 characters.
    2. Starts and ends with an alphanumeric character.
    3. Contains only alphanumeric characters, underscores, or hyphens.
    4. No consecutive periods (already covered by replacing all non-alphanumeric with underscores).
    """
    # Lowercase
    name = name.lower()
    
    # Replace any character that is not alphanumeric, underscore, or hyphen with underscore
    sanitized = re.sub(r'[^a-z0-9_-]', '_', name)
    
    # Truncate to maximum 63 characters
    sanitized = sanitized[:63]
    
    # Pad with 'a' if it is less than 3 characters
    while len(sanitized) < 3:
        sanitized += "a"
        
    # Ensure it starts with alphanumeric character
    if not sanitized[0].isalnum():
        sanitized = 'a' + sanitized[1:]
        
    # Ensure it ends with alphanumeric character
    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + 'a'
        
    return sanitized

def save_to_vector_db(documents: List[Document], repository_name: str) -> Chroma:
    """
    Saves a list of document chunks into a ChromaDB collection specific to the repository.
    Overwrites the collection if it already exists to prevent duplicate entries.
    """
    if not documents:
        raise ValueError("No supported files or content found to index. Please verify your repository/source files.")

    embeddings = get_embeddings_model()
    collection_name = sanitize_collection_name(repository_name)
    persist_directory = os.path.abspath(DB_DIR)
    
    logger.info(f"Indexing {len(documents)} chunks to ChromaDB (Collection: '{collection_name}')")
    
    # Attempt to load and delete collection first if it exists, to prevent duplicate chunks on re-processing
    try:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        db.delete_collection()
        logger.info(f"Existing collection '{collection_name}' deleted to overwrite.")
    except Exception as e:
        logger.debug(f"Could not delete collection '{collection_name}' (it may not exist yet): {e}")

    # Index documents
    try:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        logger.info(f"Successfully saved and persisted chunks to Chroma DB.")
        return db
    except Exception as e:
        logger.error(f"Failed to save documents to vector database: {e}")
        raise

def get_vector_db(repository_name: str) -> Chroma:
    """
    Loads and returns the existing ChromaDB instance for the given repository.
    """
    embeddings = get_embeddings_model()
    collection_name = sanitize_collection_name(repository_name)
    persist_directory = os.path.abspath(DB_DIR)
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

def get_retriever(repository_name: str, top_k: int = 8):
    """
    Configures and returns an MMR (Maximal Marginal Relevance) retriever for the repository.
    MMR balances relevance and diversity of retrieved source code files.
    """
    db = get_vector_db(repository_name)
    
    # Return as retriever using MMR
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": 20,       # Number of documents to fetch to pass to MMR algorithm
            "lambda_mult": 0.5   # 0.5 balances relevance and diversity
        }
    )
    return retriever

def delete_repository_index(repository_name: str) -> bool:
    """
    Deletes the vector DB collection index associated with the repository.
    """
    embeddings = get_embeddings_model()
    collection_name = sanitize_collection_name(repository_name)
    persist_directory = os.path.abspath(DB_DIR)
    
    try:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        db.delete_collection()
        logger.info(f"Successfully deleted Chroma collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.warning(f"Could not delete Chroma collection '{collection_name}': {e}")
        return False
