import tensorflow as tf
import numpy as np
from typing import Tuple, List
import yaml, time

from schema.api_schemas import Sample
from schema.api_schemas import LaneNet2Output

from helpers.preprocessing import PreProcessor
from engine import LaneNet2Engine

class LaneNet2():
    
    def __init__(
        self,
        weights_path,
        ):
        
        self.weights_path = weights_path
                                
        self.preprocessor = PreProcessor(target_size=(512, 256))
                             
        self._engine = LaneNet2Engine(weights_path)

    
    def infer(self, picture: Sample) -> LaneNet2Output:
        
        # I probably dont need the postprocessing step here since I am creating a unified preprocessing pipeline
                
        processed_image = self.preprocessor.process(picture)
        
        if processed_image.image is None:
            raise ValueError(
                f"The processed image for sample '{picture.image_path}' is None. "
                "This should not happen. Please check the preprocessing step."
            )
            
        binary_seg, instance_seg = self._engine.predict(processed_image.image)
        
        return LaneNet2Output(
            sample=picture,
            binary_segmentation=binary_seg[0],
            instance_segmentation=instance_seg[0]
        )
    
    def batch_infer(self, batch: List[Sample]) -> List[LaneNet2Output]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
    
    def close(self):
        self._engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()