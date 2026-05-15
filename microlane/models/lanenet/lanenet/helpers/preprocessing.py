from typing import Tuple, List
from schemas.api_schemas import Sample
import numpy as np
import cv2

class PreProcessor():
    
    def __init__(self) -> None:
        
        self.target_size = [256, 512]
        
        
    def process(self, sample: Sample) -> Sample:
        H, W = self.target_size
        
        image = sample.image
        
        normalized = cv2.resize(
            image,
            (W, H),
            interpolation=cv2.INTER_LINEAR # Resize and Normalize as per the original code
        ).astype(np.float32) / 127.5 - 1.0
        
        return Sample(
            image_path=sample.image_path,
            image=normalized,
            lanes=np.array(sample.lanes),
            h_samples=np.array(sample.h_samples),
            dataset=sample.dataset,
            lighting=sample.lighting,
            blur=sample.blur,
            zoom=sample.zoom,
            rotation=sample.rotation
        )