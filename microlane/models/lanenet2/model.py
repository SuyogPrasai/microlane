import tensorflow as tf
import numpy as np
from typing import Tuple, List
import yaml


from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
from microlane.schema.prediction import LanePrediction

from microlane.models.lanenet2.lanenet2.preprocessing import PreProcessor

class LaneNet2():
    
    def __init__(
        self,
        weights_path,
        config_path: str        
        ):
        
        self.weights_path = weights_path
        
        self.CFG = self._load_config(config_path)
                        
        self.preprocessor = PreProcessor(target_size=tuple(self.CFG['AUG']['EVAL_CROP_SIZE']))
                
        self._load_model()
             

    
    def infer(self, picture: Sample) -> LanePrediction:
        
        # I probably dont need the postprocessing step here since I am creating a unified preprocessing pipeline
                
        preprocessd_image = self.preprocessor.process_one(picture)
        
        
        
        
        
        
        
        
        
        return LanePrediction(
            predicted_lanes=[],
            actual_lanes=[],
            image_path="",
            image=np.zeros((1,1,3))
        )
        
    
    def batch_infer(self, batch: List[Sample]) -> List[LanePrediction]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
    
    def _load_config(self, config_path: str):
        
        # First Load the Configuation file
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
            
    
    def _load_model(self):
        
        pass