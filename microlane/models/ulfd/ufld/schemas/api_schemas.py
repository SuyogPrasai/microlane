
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Sample:
    image_path: str
    image: np.ndarray  # None until actually loaded into memory
    lanes: np.ndarray
    h_samples: np.ndarray
      
    brightness: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0
    blur: float = 0.0

    
@dataclass
class ModelPrediction:
    sample: Sample
    lanes: List[List[float]]
    run_time: float