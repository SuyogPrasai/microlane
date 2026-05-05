# Basically what this is going to do is
# Check if we have a docker container that is already running
# If we do then we use that
# If we don't then we instantiate a new docker container

# Then, we will have different functiosn for contacting the different api's of the container
# predict one, batch predict, etc

import numpy as np
import requests

from microlane.models.model import Model
from microlane.schemas.sample import Sample
from microlane.schemas.prediction import Prediction
from microlane.utils.request_processing import sample_to_payload, payload_to_prediction

class UFLD(Model):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self.model_config = self.config.models.ultra_fast_lane_detection
        
        print(self.config)
        
        self.container_id = self.initialize_model(self.model_config)

    def predict(self, sample: Sample) -> Prediction:
        
        url = f'http://localhost:{self.model_config.port}/infer'
        
        payload = sample_to_payload(sample)
        
        response = requests.post(url, json={"sample": payload})
        
        return payload_to_prediction(response.json())