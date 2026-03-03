import os


###############################################
# Directory + Path Helpers
###############################################
def ensure_directory(path: str):
    """
    Create Folder if it does not already exist.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)