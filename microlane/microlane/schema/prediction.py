
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from microlane.schema.sample import LaneLine

@dataclass
class LanePrediction:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    image_path: str
    predicted_lanes: List[LaneLine]