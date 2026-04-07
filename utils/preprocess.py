"""
Code for loading and formatting the dataset and formatting it in the right format

"""

import os, json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


class PreProcess():
    """
    Main Entry Point class for the preprocessing pipeline
    """    
    
    def __init__(self, dataset: str, model: str) -> None:
        
        self.dataset = dataset
        
        self.model = model
        
    
    def load_dataset(self):
        
        pass
    
    def format_dataset(self):
        
        pass