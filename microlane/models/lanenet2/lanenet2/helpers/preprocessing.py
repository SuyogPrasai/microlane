from typing import Tuple, List
from schemas.api_schemas import Sample
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
         
        resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 127.5 - 1.0 # type: ignore ( c++ code badly typed so error )
            
        return Sample(
            image_path=sample.image_path,
            image=resized,
            lanes=np.array(sample.lanes),
            h_samples=np.array(sample.h_samples),
            brightness=sample.brightness,
            blur=sample.blur,
            zoom=sample.zoom,
            rotation=sample.rotation
        )
