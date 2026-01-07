import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
