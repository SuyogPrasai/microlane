
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from microlane.schema.sample import LaneLine

@dataclass
class LanePrediction:
    predicted_lanes: List[LaneLine]        # predicted lanes in original image coordinates
    actual_lanes: List[LaneLine]
    image_path: str              # so you can match back to ground truth
    image: np.ndarray