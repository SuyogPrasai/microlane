"""
Code which helps us work with the TuSimple Dataset localted at data/TuSimple
"""

import os, json
from typing import Tuple, List
from microlane.schema.sample import Sample
import numpy as np
from pathlib import Path

# This is a common dataset example that can be implemented for all types, right now i have only done for TUsimple
# Buyt it may be more appropriate to create a Dataset Class and extend it for TuSimple, and add more properties accordingly

class TuSimple():    
    """
    Class structuring for the tusimple datset

    """
    
    def __init__(
        self, 
        dimensions: Tuple[int, int],
        path: str,
        annotation_file_path: str
        ) -> None:
        
        self.image_dimensions = dimensions
        self.folder_location = path
        self.annotation_file_path = annotation_file_path
        
        if(Path(self.annotation_file_path).exists() is False):
            raise FileNotFoundError("The annotation file path provided does not exist")
        
    
    def load(self, number = 300) -> List[Sample]:
        samples = []
        
        with open(self.annotation_file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= number:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                
                samples.append(data)

        return samples
    
    
    def load_image(self, sample: Sample) -> np.ndarray:
        raise NotImplementedError("load_image is not implemented yet")