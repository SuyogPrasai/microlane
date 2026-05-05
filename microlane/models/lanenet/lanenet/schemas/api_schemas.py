
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Sample:
    """
    Represents a common sample for all datasets,
    Process each image from the dataset and contruct a common sample which can be fed into any model
    
    """
    
    image_path: str
    image: np.ndarray
    lanes: np.ndarray # x values
    h_samples: np.ndarray # y values
    dataset: str
    
    # Image Augmentation features
    blur: float = 0.0
    lighting: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0
    motion_blur: float = 0.0
    
    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def __setitem__(self, key: str, value):
        return setattr(self, key, value)
    
@dataclass
class Prediction:
    """
    Represents a common prediction for all models,
    Process each model's output and contruct a common prediction which can be evaluated against the ground truth sample
    
    """
    samples: List[Sample]
    lanes: np.ndarray # x values
    h_samples: np.ndarray # y values
    run_time: float