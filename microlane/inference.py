"""
Code for doing the actual inference, need to handle for one image and batch images
And after that output the result in a format that we can use

"""

import os, json


class  Inference():
    
    
    # This is the main inferencing class, which will have all the helper functions and properties to do the inferencing
    def __init__(self, model: str, dataset: str) -> None:
        
        self.model = model
        
        self.dataset = dataset
        