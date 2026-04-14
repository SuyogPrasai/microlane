
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

    def __getitem__(self, key: str):
        return getattr(self, key)