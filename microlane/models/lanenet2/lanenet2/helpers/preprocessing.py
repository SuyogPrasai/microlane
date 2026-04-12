from typing import Tuple, List
from schema.api_schemas import Sample
import numpy as np
import cv2


class PreProcessor:
    
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size
        
    def process(self, sample: Sample) -> Sample:
        if sample.image is None:
            raise ValueError(
                f"Sample '{sample.image_path}' has no loaded image. "
                "Call sample.load() before formatting."
            )
 
        W, H = self.target_size
        image: np.ndarray = sample.image
         
        resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 127.5 - 1.0
 
        return Sample(
            image=resized,
            image_path=sample.image_path,
            actual_lanes=sample.actual_lanes,
            modified=True
        )