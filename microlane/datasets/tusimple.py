"""
Code which helps us work with the TuSimple Dataset localted at data/TuSimple
"""

import os, json
from typing import Tuple, List
from microlane.schema.sample import Sample, LaneLine
import numpy as np
from pathlib import Path
import cv2

# This is a common dataset example that can be implemented for all types, right now i have only done for TUsimple
# Buyt it may be more appropriate to create a Dataset Class and extend it for TuSimple, and add more properties accordingly

class TuSimple():    
    """
    Class structuring for the tusimple datset

    """
    
    def __init__(
        self, 
        annotation_file_path: str,
        folder_path: str
        ) -> None:
        
        self.annotation_file_path = annotation_file_path
        self.folder_path = folder_path
        
        if(Path(self.annotation_file_path).exists() is False):
            raise FileNotFoundError("The annotation file path provided does not exist")
        
    
    def load(self, number=300) -> List[Sample]:
        samples = []

        with open(self.annotation_file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= number:
                    break

                line = line.strip()
                if not line:
                    continue

                data: dict = json.loads(line)

                lanes = [
                    LaneLine(
                        x_coordinates=lane_xs,
                        y_coordinates=data["h_samples"],
                    )
                    for lane_xs in data["lanes"]
                ]

                sample = Sample(
                    image_path=self.folder_path + data["raw_file"],
                    image=None,
                    actual_lanes=lanes,
                )

                samples.append(sample)

        return samples
    
    def load_image(self, sample: Sample) -> Sample:
        image = cv2.imread(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at '{sample.image_path}'")
        sample.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return sample
    
    def load_images(self, samples: List[Sample]) -> List[Sample]:
        return [self.load_image(sample) for sample in samples]