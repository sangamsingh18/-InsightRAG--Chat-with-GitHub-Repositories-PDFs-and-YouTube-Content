import os
import logging
from pypdf import PdfReader
from langchain_core.documents import Document

# Setup logging
logger = logging.getLogger(__name__)

def load_uploaded_file(uploaded_file) -> list[Document]:
    """
    Parses a Streamlit UploadedFile object based on its file extension and returns a Document.
    
    Supported formats:
    - PDF: Extracts text page by page (adding page headers).
    - Text formats (.txt, .md, .csv, .json, .yaml, .yml, .sql, .py, etc.): Decodes text.
    """
    name = uploaded_file.name
    _, ext = os.path.splitext(name.lower())
    
    logger.info(f"Loading uploaded file: '{name}' (Extension: '{ext}')...")
    
    content = ""
    try:
        if ext == ".pdf":
            # Extract PDF content
            pdf_bytes = uploaded_file.read()
            # Feed file-like bytes into PdfReader
            import io
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            
            pages_text = []
            for idx, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(f"\n--- Document Page {idx+1} ---\n{page_text.strip()}\n")
            content = "".join(pages_text)
            
        else:
            # General text decode fallback
            file_bytes = uploaded_file.read()
            content = file_bytes.decode("utf-8", errors="ignore")
            
        if not content.strip():
            raise ValueError(f"No readable text content extracted from file '{name}'.")
            
        metadata = {
            "source": "document",
            "file_name": name,
            "file_path": name,
            "language": ext.lstrip(".").upper() if ext else "TXT"
        }
        
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        logger.info(f"Successfully loaded uploaded file '{name}'.")
        return [doc]
        
    except Exception as e:
        logger.error(f"Error loading uploaded file '{name}': {e}")
        raise RuntimeError(f"Failed to read file '{name}': {str(e)}")
