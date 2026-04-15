from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from microlane.schema.sample import Sample
 
@dataclass
class ModelPrediction:
    sample: Sample
    lanes: List[List[float]]
    inference_time: float