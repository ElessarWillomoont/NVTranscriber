import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.mp4', '.mkv', '.mp3', '.m4a', '.wav'}

class Scanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        
    def scan(self):
        """Recursively scan for supported media files."""
        found_files = []
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            return found_files
            
        for ext in SUPPORTED_EXTENSIONS:
            found_files.extend([str(p) for p in self.root_dir.rglob(f"*{ext}")])
            
        # Also need to match uppercase extensions
        for ext in SUPPORTED_EXTENSIONS:
            found_files.extend([str(p) for p in self.root_dir.rglob(f"*{ext.upper()}")])
            
        return list(set(found_files))
