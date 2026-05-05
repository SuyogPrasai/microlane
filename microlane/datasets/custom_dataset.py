from microlane.schemas.sample import Sample
from typing import List
from pathlib import Path


class CustomDataset():
    
    def __init__(
        self, 
        folder_path: Path, 
        annotation_file_path: Path) -> None:
        
        self.folder_path = folder_path
        self.annotation_file_path = annotation_file_path
        
        # Add checks and verifications
        if self.annotation_file_path.exists() is False:
            raise FileNotFoundError(f"Annotation file not found at {self.annotation_file_path}")
        
        if self.folder_path.exists() is False:
            raise FileNotFoundError(f"Folder path not found at {self.folder_path}")
    
        if self.folder_path.is_dir() is False:
            raise NotADirectoryError(f"Provided folder path is not a directory: {self.folder_path}")
        
        if self.annotation_file_path.is_file() is False:
            raise ValueError(f"Provided annotation path is not a file: {self.annotation_file_path}")
        
        if self.annotation_file_path.suffix.lower() != '.xml':
            raise ValueError(f"Annotation file must be an XML file: {self.annotation_file_path}")


    def load(self, number: int = 500) -> List[Sample]:
        
        samples: List[Sample] = []
        
        # Let's Directly convert the XML annotations to Sample objects here
        
        return samples