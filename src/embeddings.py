import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings model using the BAAI/bge-small-en-v1.5 model.
    Automatically detects and utilizes GPU (CUDA) if available.
    """
    model_name = "BAAI/bge-small-en-v1.5"
    
    # Check if GPU is available without forcing torch to be a hard import unless needed
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        logger.info("Torch not installed or could not check CUDA. Defaulting embeddings to CPU.")

    logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
    
    # Configure model and encoding arguments
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}  # Normalize embeddings for cosine similarity
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise
