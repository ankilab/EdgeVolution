import os

def find_project_root(project_name="EdgeVolution"):
    """
    Find the project root directory by moving up in the directory hierarchy
    until the specified project directory is found (case-insensitive).
    
    Args:
        project_name: Name of the project directory to find
    
    Returns:
        The absolute path to the project root directory, or current directory if not found
    """
    folder = os.getcwd()
    max_depth = 10  # Prevent infinite loop
    depth = 0
    root_folder = os.path.abspath(os.sep)
    
    while depth < max_depth and folder != root_folder:
        if os.path.basename(folder).lower() == project_name.lower():
            break
        folder = os.path.dirname(folder)
        depth += 1
    
    # Fallback if directory not found
    if depth >= max_depth or folder == root_folder:
        print(f"Warning: '{project_name}' directory not found, using current directory")
        folder = os.getcwd()
    
    return folder
