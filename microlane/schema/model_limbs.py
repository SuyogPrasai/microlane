from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# formatters/ufld.py

@dataclass  
class UFLDInput:
    image: np.ndarray        # (H, W, 3) normalized
    cls_label: np.ndarray    # (num_lanes, num_row_anchors) classification targets