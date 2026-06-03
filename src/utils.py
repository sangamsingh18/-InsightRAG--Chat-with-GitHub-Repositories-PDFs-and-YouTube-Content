import logging
import tiktoken

# Setup logging
logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a string using tiktoken.
    Falls back to a character length heuristic if tiktoken fails.
    """
    try:
        # Use cl100k_base encoding (common for Llama/OpenAI models)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.debug(f"Failed to use tiktoken for token counting: {e}")
        # Fallback approximation: 1 token ~ 4 characters
        return len(text) // 4

def build_ascii_tree(node: dict, prefix: str = "", is_last: bool = True) -> str:
    """
    Recursively builds a formatted ASCII text representation of the repository structure.
    
    Args:
        node: A node dictionary from repo_parser.py
        prefix: Current line indentation prefix.
        is_last: Boolean indicating if this is the last sibling at the current level.
        
    Returns:
        A multiline ASCII tree string.
    """
    result = ""
    
    # Root node formatting
    if prefix == "":
        result += f"📁 {node['name']}/\n"
    else:
        connector = "└── " if is_last else "├── "
        icon = "📁 " if node['type'] == 'directory' else "📄 "
        
        # Add details for files
        details = ""
        if node['type'] == 'file':
            lang = node.get('language', '')
            lines = node.get('lines', 0)
            details = f" ({lang}, {lines} lines)"
            
        result += f"{prefix}{connector}{icon}{node['name']}{details}\n"
        
    if "children" in node:
        # Determine the prefix to pass to child elements
        # If the current directory is the last item, we indent without vertical lines.
        # Otherwise, we append a vertical pipe.
        next_prefix = prefix
        if prefix != "":
            next_prefix += "    " if is_last else "│   "
        else:
            # Root children don't need root indent
            next_prefix = ""
            
        children = node["children"]
        for idx, child in enumerate(children):
            is_child_last = (idx == len(children) - 1)
            result += build_ascii_tree(child, next_prefix, is_child_last)
            
    return result

def clean_relative_path(path: str) -> str:
    """
    Normalizes path formatting, replacing backslashes with forward slashes.
    """
    return path.replace('\\', '/')

def log_indexed_source(source_type: str, details: str, filepath: str = "indexed_sources.txt") -> None:
    """
    Logs any indexed source (GitHub link, YouTube link, or Document details) 
    to the Supabase database if configured, otherwise falls back to a local text log file.
    """
    import os
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    # 1. Try logging to Supabase if configured
    if url and key and url.strip() != "" and key.strip() != "":
        try:
            from supabase import create_client, Client
            supabase: Client = create_client(url, key)
            
            # Check for duplicates to prevent unique constraint errors
            response = supabase.table("indexed_sources") \
                .select("id") \
                .eq("source_type", source_type) \
                .eq("source_url", details.strip()) \
                .execute()
                
            if response.data:
                logger.info(f"Source already logged in Supabase: [{source_type}] {details}")
                return
                
            supabase.table("indexed_sources").insert({
                "source_type": source_type,
                "source_url": details.strip(),
                "details": f"Successfully indexed into vector DB."
            }).execute()
            logger.info(f"Successfully logged source to Supabase: [{source_type}] {details}")
            return
        except Exception as e:
            logger.error(f"Supabase logging failed: {e}. Falling back to local file logging.")
            # Fall through to local file logging in case of error

    # 2. Local Text File Fallback
    log_entry = f"[{source_type}] {details.strip()}"
    
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# InsightRAG Indexed Sources Log\n")
            f.write("# This file tracks all indexed knowledge bases sequentially.\n\n")
            f.write(f"1. {log_entry}\n")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Prevent duplicates
    for line in lines:
        if log_entry in line:
            return

    # Count actual numbered items (ignoring headers)
    item_count = 0
    for line in lines:
        line_strip = line.strip()
        if line_strip and not line_strip.startswith("#"):
            item_count += 1

    serial = item_count + 1
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{serial}. {log_entry}\n")

def safe_rmtree(path: str) -> None:
    """
    Safely deletes a directory tree, resolving read-only file permission issues on Windows.
    """
    import os
    import stat
    import shutil

    def remove_readonly(func, file_path, excinfo):
        try:
            os.chmod(file_path, stat.S_IWRITE)
            func(file_path)
        except Exception:
            pass

    if os.path.exists(path):
        shutil.rmtree(path, onerror=remove_readonly)
