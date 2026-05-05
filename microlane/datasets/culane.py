from microlane.schemas.sample import Sample
from typing import List
from pathlib import Path

class CuLane:
    
    def __init__(
        self, 
        folder_path: Path) -> None:
        
        self.folder_path = folder_path
        

        if self.folder_path.exists() is False:
            raise FileNotFoundError(f"Folder path not found at {self.folder_path}")
    
        if self.folder_path.is_dir() is False:
            raise NotADirectoryError(f"Provided folder path is not a directory: {self.folder_path}")
        

    def load_samples(self) -> List[Sample]:
        
        samples: List[Sample] = []
        
        # Let's Directly convert the XML annotations to Sample objects here
        
        return samples