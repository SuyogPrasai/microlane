from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from microlane.schema.sample import Sample

@dataclass
class LaneNet2Output:
    sample: Sample
    binary_segmentation: np.ndarray  # (H, W) binary mask of lane pixels
    instance_segmentation: Optional[np.ndarray] = None  # (H, W) instance