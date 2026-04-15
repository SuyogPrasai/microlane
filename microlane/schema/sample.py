
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class LaneLine:
    x_coordinates: List[float]   # x positions, -2 means lane doesn't exist at that y
    y_coordinates: List[float]   # the fixed row anchors (same for all lanes in TuSimple)
    
@dataclass
class Sample:
    image_path: str
    image: Optional[np.ndarray]  # None until actually loaded into memory
    actual_lanes: List[LaneLine] 
    modified: bool = False
    
    blur: float = 0.0
    brightness: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def __setitem__(self, key: str, value):
        return setattr(self, key, value)