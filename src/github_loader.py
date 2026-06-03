import os
import re
import shutil
import logging
from git import Repo
from src.utils import safe_rmtree

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_github_url(url: str):
    """
    Validates and parses a GitHub URL.
    Returns (owner, repo, clean_url) or (None, None, None)
    
    Supports:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - http://github.com/owner/repo
    """
    # Normalize url (strip whitespace and trailing slashes)
    url_stripped = url.strip().rstrip("/")
    
    # Regex to match github url
    pattern = r"^https?://(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?$"
    match = re.match(pattern, url_stripped, re.IGNORECASE)
    
    if not match:
        return None, None, None
        
    owner = match.group(1)
    repo = match.group(2)
    clean_url = f"https://github.com/{owner}/{repo}.git"
    return owner, repo, clean_url

def clone_repository(url: str, base_dir: str = "repositories", refresh: bool = False) -> str:
    """
    Clones a public GitHub repository locally.
    
    Args:
        url: The public GitHub repository URL.
        base_dir: The local base directory to clone repositories into.
        refresh: If True, pulls updates or re-clones if existing.
        
    Returns:
        The absolute path to the cloned repository.
    """
    owner, repo, clean_url = parse_github_url(url)
    if not owner or not repo:
        raise ValueError(
            f"Invalid GitHub URL: '{url}'. Please provide a valid public repository URL (e.g., https://github.com/username/project)."
        )

    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    repo_dir_name = f"{owner}_{repo}"
    repo_path = os.path.abspath(os.path.join(base_dir, repo_dir_name))

    # Check if directory exists and is a valid Git repository (preventing parent directory traversal)
    is_valid_git_repo = False
    if os.path.exists(repo_path):
        try:
            Repo(repo_path, search_parent_directories=False)
            is_valid_git_repo = True
        except Exception:
            is_valid_git_repo = False

    if os.path.exists(repo_path):
        if is_valid_git_repo and not refresh:
            logger.info(f"Repository already exists locally and is valid at: {repo_path}")
            return repo_path
        
        if refresh and is_valid_git_repo:
            logger.info(f"Repository folder exists and is valid. Attempting pull to refresh: {repo_path}")
            try:
                git_repo = Repo(repo_path, search_parent_directories=False)
                origin = git_repo.remotes.origin
                origin.pull()
                logger.info("Successfully pulled latest changes.")
                return repo_path
            except Exception as e:
                logger.warning(f"Git pull failed: {e}. Falling back to full re-clone.")
                safe_rmtree(repo_path)
        else:
            logger.warning(f"Repository path exists but is not a valid git repository or force re-clone requested. Re-cloning: {repo_path}")
            safe_rmtree(repo_path)

    # Make sure target path doesn't exist before cloning
    if os.path.exists(repo_path):
        try:
            safe_rmtree(repo_path)
        except Exception as e:
            logger.error(f"Error removing existing path {repo_path}: {e}")
            raise

    logger.info(f"Cloning {clean_url} (depth=1) into {repo_path}...")
    try:
        Repo.clone_from(clean_url, repo_path, depth=1)
        logger.info("Cloning completed successfully.")
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        # Clean up any partial clone
        if os.path.exists(repo_path):
            safe_rmtree(repo_path)
        raise RuntimeError(f"Cloning failed: {e}")

    return repo_path
