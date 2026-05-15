# Interface for LaneNet model in the pipeline

import requests
from typing import List

from microlane.models.model import Model
from microlane.schemas.sample import Sequence
from microlane.schemas.prediction import Prediction
from microlane.utils.request_processing import samples_to_payload, payload_to_prediction

class RLD(Model):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self.model_config = self.config.models.rld_a
                
        self.container_id = self.initialize_model(self.model_config)
    

    def predict(self, sequence: Sequence) -> Prediction:
        
        url = f'http://localhost:{self.model_config.port}/infer'
        
        payload = samples_to_payload(sequence.samples)
        
        response = requests.post(url, json={"samples": payload})
        
        return payload_to_prediction(response.json())