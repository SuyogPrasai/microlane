
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
    
@dataclass
class Sample:
    image_path: str
    image: Optional[np.ndarray]  # None until actually loaded into memory
    lanes: List[List[float]]
    h_samples: List[float]
    
    blur: float = 0.0
    brightness: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0
    motion_blur: float = 0.0

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def __setitem__(self, key: str, value):
        return setattr(self, key, value)