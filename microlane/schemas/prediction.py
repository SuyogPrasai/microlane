
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from microlane.schemas.sample import Sample

@dataclass
class Prediction:
    """
    Represents a common prediction for all models,
    Process each model's output and contruct a common prediction which can be evaluated against the ground truth sample
    
    """
    samples: List[Sample]
    lanes: np.ndarray # x values
    h_samples: np.ndarray # y values
    run_time: float
    
    
@dataclass
class Evaluation:
    """
    Represents a common evaluation for all models,
    Process each model's prediction and contruct a common evaluation which can be compared across models
    
    """
    predictions: Prediction
    accuracy: float
    precision: float
    recall: float
    f1_score: float