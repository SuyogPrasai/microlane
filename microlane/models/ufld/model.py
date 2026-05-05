# Basically what this is going to do is
# Check if we have a docker container that is already running
# If we do then we use that
# If we don't then we instantiate a new docker container

# Then, we will have different functiosn for contacting the different api's of the container
# predict one, batch predict, etc

import requests

from microlane.models.container import ContainerManager
from microlane.schema.sample import Sample
from microlane.schema.output import ModelPrediction

class UFLD():
    
    def __init__(self, container_folder: str, image_name: str, port = 8000) -> None:
        
        self.container_port = port
        
        self.container_folder = container_folder
        
        self.image_name = image_name

        self.container_manager = ContainerManager(
            
            port=self.container_port, 
            
            container_folder=self.container_folder,
            
            image_name=self.image_name
        )
        
        self.container_manager.initialize_container()

    def predict(self, sample: Sample):
        
        url = f'http://localhost:{self.container_port}/infer'
        
        payload = self.sample_to_payload(sample)
        
        response = requests.post(url, json={"sample": payload})
        
        return response
    
    
    def sample_to_payload(self, sample: Sample) -> dict:
        
        return {
            "image_path": sample.image_path,
            
            "lanes": sample.lanes,
            "h_samples": sample.h_samples,
            
            "image": sample.image.tolist() if sample.image is not None else None,  # ndarray → nested list,
            "blur": sample.blur,
            "lighting": sample.lighting,
            "zoom": sample.zoom,
            "rotation": sample.rotation,
        }