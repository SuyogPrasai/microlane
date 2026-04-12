import tensorflow as tf
import numpy as np
from typing import Tuple, List
import yaml, time

from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
from microlane.schema.prediction import LanePrediction

from microlane.models.lanenet2.lanenet2.helpers.preprocessing import PreProcessor

class LaneNet2():
    
    def __init__(
        self,
        weights_path,
        ):
        
        self.weights_path = weights_path
                                
        self.preprocessor = PreProcessor(target_size=(512, 256))
                             

    
    def infer(self, picture: Sample) -> LanePrediction:
        
        # I probably dont need the postprocessing step here since I am creating a unified preprocessing pipeline
                
        preprocessd_image = self.preprocessor.process_one(picture)
        
        return LanePrediction(
            binary_segmentation=np.zeros((256, 512))
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
        