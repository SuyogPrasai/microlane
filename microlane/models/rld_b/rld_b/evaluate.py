from typing import List
import numpy as np # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]

from schemas.api_schemas import Sample, Prediction

from engine import RLDEngine

from helpers.preprocessing import PreProcessor
from helpers.postprocessing import PostProcessor

class RLD():
    
    def __init__(self, weights_path) -> None:
        
        self.use_gpu = True
        
        self.model_name = "SegNet-ConvLSTM"
        
        self.weights_path = weights_path
        
        self.preprocessor = PreProcessor()
        
        self.postprocessor = PostProcessor()
        
        self.engine = RLDEngine(
            
            model_name=self.model_name,
            weights_path=self.weights_path,
            use_gpu=self.use_gpu
        )
        
    
    def infer(self, images: List[Sample]) -> Prediction:
        
        if not images:
            raise ValueError("images must contain at least one Sample")
        
        input_tensors = self.preprocessor.process(
         
            samples=images,
        )
        
        mask, run_time = self.engine.predict(input_tensors)
        
        return self.postprocessor.process(
            samples=images,
            run_time=run_time,
            mask=mask
        )

        
        