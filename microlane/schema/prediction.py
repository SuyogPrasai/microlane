
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from microlane.schema.sample import LaneLine

@dataclass
class LanePrediction:
    lanes: List[LaneLine]        # predicted lanes in original image coordinates
    confidence: List[float]      # one score per lane
    image_path: str              # so you can match back to ground truth