import tensorflow as tf
import numpy as np
from typing import Tuple, List

from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
from microlane.schema.prediction import LanePrediction

class LaneNet2():
    
    def __init__(
        self,
        weights_path,        
        ):
        
        self.weights_path = weights_path
        
        self._load_model()
             

    
    def infer(self, image: LaneNet2Input) -> LanePrediction:
           
        return LanePrediction(
            lanes=[],
            confidence=[],
            image_path="",
        )
        
    
    def batch_infer(self, batch: List[LaneNet2Input]) -> List[LanePrediction]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
    
    
    def _load_model(self):
        
        pass