import os
import logging
from typing import Dict, List, Any, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Extensions with leading dot, plus specific exact filenames
SUPPORTED_EXTENSIONS = {
    # Code extensions
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', 
    # Markup and configs
    '.html', '.css', '.scss', '.json', '.yaml', '.yml', '.md', '.txt', '.sql'
}

SUPPORTED_FILENAMES = {
    'dockerfile', 'package.json', 'requirements.txt', 'makefile', 'jenkinsfile', '.gitignore'
}

IGNORED_DIRS = {
    'node_modules', 'venv', '.git', 'dist', 'build', 'coverage', 
    '__pycache__', '.pytest_cache', '.idea', '.vscode', '.github',
    'bin', 'obj', 'vendor'
}

def get_language(filename: str) -> str:
    """
    Maps file names or extensions to a human-readable programming language name.
    """
    fn_lower = filename.lower()
    
    # Specific filename matches
    if fn_lower == 'package.json':
        return 'Node Package Configuration'
    if fn_lower == 'requirements.txt':
        return 'Python Requirements'
    if fn_lower == 'dockerfile' or fn_lower.endswith('.dockerfile'):
        return 'Dockerfile'
    if fn_lower == 'makefile':
        return 'Makefile'
    if fn_lower == '.gitignore':
        return 'Git Configuration'
    
    _, ext = os.path.splitext(fn_lower)
    
    ext_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript React',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript React',
        '.java': 'Java',
        '.cpp': 'C++',
        '.hpp': 'C++ Header',
        '.c': 'C',
        '.h': 'C/C++ Header',
        '.cs': 'C#',
        '.go': 'Go',
        '.rs': 'Rust',
        '.php': 'PHP',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.md': 'Markdown',
        '.txt': 'Text',
        '.sql': 'SQL'
    }
    return ext_map.get(ext, 'Text/Config')

def is_supported(filename: str) -> bool:
    """
    Checks if a file is supported based on its name or extension.
    """
    fn_lower = filename.lower()
    _, ext = os.path.splitext(fn_lower)
    
    # Check if file name or extension is explicitly supported
    if fn_lower in SUPPORTED_FILENAMES or ext in SUPPORTED_EXTENSIONS:
        return True
    # Catch wildcard Dockerfile names (e.g. Dockerfile.dev)
    if 'dockerfile' in fn_lower:
        return True
    return False

def parse_repository(repo_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Walks the directory recursively, loads supported file contents, builds directory tree representation,
    and calculates repository statistics.
    
    Returns:
        parsed_files: List of file descriptions containing metadata and text contents.
        stats: Dictionary containing files/folders counts, total line numbers, and languages counts.
        tree: Nested dictionary representing the repository's file structure.
    """
    parsed_files = []
    stats = {
        "total_files": 0,
        "total_folders": 0,
        "languages": {},
        "total_lines": 0,
        "total_bytes": 0
    }
    
    repo_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(repo_path)
    
    # File structure tree representation
    tree = {"name": repo_name, "type": "directory", "children": [], "path": ""}
    path_to_node = {"": tree}
    
    unique_folders = set()
    
    for root, dirs, files in os.walk(repo_path):
        # Prune ignored folders in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith('.')]
        
        # Calculate relative path from the repository root
        rel_root = os.path.relpath(root, repo_path)
        if rel_root == ".":
            rel_root = ""
            
        # Create folder nodes in the tree structure
        for d in dirs:
            rel_dir_path = os.path.join(rel_root, d).replace('\\', '/') if rel_root else d
            unique_folders.add(rel_dir_path)
            
            # Find the parent node
            parent_node = path_to_node[rel_root.replace('\\', '/')]
            dir_node = {"name": d, "type": "directory", "children": [], "path": rel_dir_path}
            parent_node["children"].append(dir_node)
            path_to_node[rel_dir_path] = dir_node
            
        for file in files:
            if not is_supported(file):
                continue
                
            rel_file_path = os.path.join(rel_root, file).replace('\\', '/') if rel_root else file
            abs_file_path = os.path.join(root, file)
            
            # Read file contents
            try:
                # Use errors="ignore" to read text files that might contain invalid unicode
                with open(abs_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read file {abs_file_path}: {e}")
                continue
                
            lines_count = len(content.splitlines())
            file_bytes = len(content.encode("utf-8", errors="ignore"))
            lang = get_language(file)
            
            file_info = {
                "file_name": file,
                "file_path": rel_file_path,
                "absolute_path": abs_file_path,
                "language": lang,
                "content": content,
                "lines_count": lines_count,
                "size_bytes": file_bytes,
                "repository_name": repo_name
            }
            parsed_files.append(file_info)
            
            # Update stats
            stats["total_files"] += 1
            stats["total_lines"] += lines_count
            stats["total_bytes"] += file_bytes
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            # Append file to the tree
            parent_node = path_to_node[rel_root.replace('\\', '/')]
            file_node = {
                "name": file,
                "type": "file",
                "path": rel_file_path,
                "language": lang,
                "lines": lines_count,
                "size": file_bytes
            }
            parent_node["children"].append(file_node)
            
    stats["total_folders"] = len(unique_folders)
    
    # Sort children in the tree so directories appear first, then files alphabetically
    def sort_tree(node):
        if "children" in node:
            node["children"].sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))
            for child in node["children"]:
                sort_tree(child)
                
    sort_tree(tree)
    
    return parsed_files, stats, tree
