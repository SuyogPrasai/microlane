from dataclasses import dataclass
from typing import List, Optional

    
@dataclass
class Evaluation:
    """
    Represents the evluation row for every prediction
    that is to be stored in a raw CSV file
    
    """
    experiment_number: int
    dataset: str
    model: str
    augmentation: str
    raw_file: str
    processed_samples: List[str]
    run_time: float
    accuracy: float
    IOU: float
    fn: float
    fp: float
    
@dataclass
class PredictionFile:
    file: List
    model: str
    dataset: str
    augmentation: dict