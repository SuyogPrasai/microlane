from schemas.api_schemas import Sample
from schemas.api_schemas import ModelPrediction

from helpers.preprocessing import PreProcessor
from engine import UFLDEngine

class UFLD():
    
    def __init__(self, weights_path) -> None:
        
        self._engine = UFLDEngine(weights_path)

    
    def infer(self, picture: Sample ) -> ModelPrediction:
        
        pass
    
    def close(self):
        self._engine.close()
    
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()