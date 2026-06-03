from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_files(
    parsed_files: List[Dict[str, Any]], 
    chunk_size: int = 1500, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits file content into overlapping chunks and formats them as LangChain Documents.
    
    Args:
        parsed_files: List of file dictionaries from the repository parser.
        chunk_size: Target character length of each chunk.
        chunk_overlap: Target overlap between sequential chunks.
        
    Returns:
        A list of LangChain Document objects ready for embedding generation.
    """
    # Use RecursiveCharacterTextSplitter suitable for code and text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""]
    )
    
    documents = []
    
    for file_info in parsed_files:
        content = file_info["content"]
        if not content.strip():
            continue
            
        # Split text into chunks
        chunks = splitter.split_text(content)
        
        for idx, chunk_text in enumerate(chunks):
            # Prepend metadata details directly to the chunk text. 
            # This ensures that even when retrieved standalone, the LLM has context of which file and language the snippet belongs to.
            contextualized_text = f"File: {file_info['file_path']}\nLanguage: {file_info['language']}\n\n{chunk_text}"
            
            metadata = {
                "file_name": file_info["file_name"],
                "file_path": file_info["file_path"],
                "language": file_info["language"],
                "repository_name": file_info["repository_name"],
                "chunk_index": idx
            }
            
            doc = Document(
                page_content=contextualized_text,
                metadata=metadata
            )
            documents.append(doc)
            
    return documents
