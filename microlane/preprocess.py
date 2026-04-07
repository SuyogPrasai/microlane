"""
Code for loading and formatting the dataset and formatting it in the right format

"""

import os, json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from microlane.datasets.tusimple import TuSimple

class PreProcess():
    """
    Main Entry Point class for the preprocessing pipeline
    """    
    
    def __init__(self, dataset: str, model: str) -> None:
        
        self.dataset = dataset
        
        self.model = model
        
    # There would be common dataset format that would work across all types of datasets like TuSimple or CuLane
    # This would return dataset format which we can feed to inferencing class
    def load_dataset(self):
        
        pass
    
    
    # There would make the formatting nescessary for each individual model needs, like for LaneNet, UFLD
    def format_dataset(self):
        
        pass