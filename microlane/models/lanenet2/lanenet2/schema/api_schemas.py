
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
    
@dataclass
class LaneNet2Output:
    sample: Sample
    binary_segmentation: np.ndarray  # (H, W) binary mask of lane pixels
    instance_segmentation: Optional[np.ndarray] = None  # (H, W) instance