"""
Code for doing the actual inference, need to handle for one image and batch images
And after that output the result in a format that we can use

"""

import os, json


class  Inference():
    
    def __init__(self, model: str, dataset: str) -> None:
        
        self.model = model
        
        self.dataset = dataset
        