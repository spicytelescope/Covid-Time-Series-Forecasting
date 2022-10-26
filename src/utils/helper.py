from pathlib import Path

def get_project_root() -> Path:
    """return the root's folder path

    Returns:
        Path: path to the root folder
    """
    return Path(__file__).parent.parent.parent
