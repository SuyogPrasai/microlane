from typing import Tuple, List
from schema.api_schemas import Sample
from schema.api_schemas import LaneNet2InputSingle
import numpy as np
import cv2


class PreProcessor:
    
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size
        
    def process_one(self, sample: Sample) -> LaneNet2InputSingle:
        if sample.image is None:
            raise ValueError(
                f"Sample '{sample.image_path}' has no loaded image. "
                "Call sample.load() before formatting."
            )
 
        W, H = self.target_size
        image: np.ndarray = sample.image
         
        resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 127.5 - 1.0
 
        return LaneNet2InputSingle(
            image=resized,
            lanes=sample.actual_lanes
        )
    
    def batch_process(self, samples: List[Sample]) -> List[LaneNet2InputSingle]:
        return [self.process_one(sample) for sample in samples]