from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# formatters/lanenet.py

@dataclass
class LaneNet2Input:
    image: np.ndarray        # (H, W, 3) normalized
    
# formatters/ufld.py

@dataclass  
class UFLDInput:
    image: np.ndarray        # (H, W, 3) normalized
    cls_label: np.ndarray    # (num_lanes, num_row_anchors) classification targets