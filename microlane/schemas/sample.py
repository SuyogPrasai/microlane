
from dataclasses import dataclass
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
    lanes: np.ndarray # x values, 2D list
    h_samples: np.ndarray # y values 1D list
    dataset: str
    
    # Image Augmentation features
    blur: float = 0.0
    lighting: float = 0.0
    rotation: float = 0.0
    zoom: float = 1.0
    motion_blur: float = 0.0
    
@dataclass
class Sequence:
    samples: List[Sample]