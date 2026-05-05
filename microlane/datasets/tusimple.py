from pathlib import Path
from typing import List
import cv2, json
import numpy as np

from microlane.schemas.sample import Sample

class TuSimple():
    
    def __init__(
        self,
        folder_path: Path,
        annotation_file_path: Path
        ) -> None:
        
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
        
        if self.annotation_file_path.suffix.lower() != '.json':
            raise ValueError(f"Annotation file must be a JSON file: {self.annotation_file_path}")
        
    
    def load(self, number: int = 500) -> List[Sample]:
        
        samples: List[Sample] = []
        
        with open(self.annotation_file_path, 'r') as f:
            
            for i, line in enumerate(f):
                if i >= number:
                    break
                
                data = json.loads(line)
                
                image_path = self.folder_path / data['raw_file']
                
                if not image_path.exists():    
                    print(f"Warning: Image file not found at [{image_path}], skipping sample.")
                    continue
                
                image = cv2.imread(str(image_path))
                
                if image is None:
                    print(f"Warning: Failed to load image at [{image_path}], skipping sample.")
                    continue
                
                samples.append(
                    Sample(
                        image_path=str(image_path),
                        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        lanes=np.array(data['lanes']),
                        h_samples=np.array(data['h_samples']),
                        dataset="TuSimple"
                    )
                )
        return samples
    
    def sample(self, raw_file: Path) -> Sample:
        
        with open(self.annotation_file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                image_path = self.folder_path / data['raw_file']
                
                if image_path != raw_file:
                    continue
                
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found at [{image_path}]")
                
                image = cv2.imread(str(image_path))
                
                if image is None:
                    raise ValueError(f"Failed to load image at [{image_path}]")
                
                return Sample(
                    image_path=str(image_path),
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    lanes=np.array(data['lanes']),
                    h_samples=np.array(data['h_samples']),
                    dataset="TuSimple"
                )
        
        raise RuntimeError(f"Sample with raw_file '{raw_file}' not found in annotation file")
        