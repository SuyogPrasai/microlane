from typing import Tuple, List
from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
from microlane.schema.prediction import LanePrediction

class LaneNet2():
    
    def __init__(self):
        
        pass
    
    def infer(self, image: LaneNet2Input) -> LanePrediction:
           
        pass
    
    def batch_infer(self, batch: List[LaneNet2Input]) -> List[LanePrediction]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
