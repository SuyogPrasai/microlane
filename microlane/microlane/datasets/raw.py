
"""
Code which helps us work with the Raw Dataset with only the Raw Image
"""

import os, json
from typing import Tuple, List
from microlane.schema.sample import Sample
import numpy as np
from pathlib import Path
import cv2

TUSIMPLE_H_SAMPLES=[160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]


class Raw():    
    """
    Class structuring for the tusimple datset

    """
    
    def __init__(
        self, 
        folder_path: str,
        annotation: bool = False,
        annotation_file_path: str = "",
        ) -> None:
        
        self.annotation_file_path = annotation_file_path
        self.folder_path = folder_path
        
        if(Path(self.annotation_file_path).exists() is False) and annotation:
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

                sample = Sample(
                    image_path=data["raw_file"],
                    lanes=data["lanes"],
                    h_samples=data["h_samples"],
                    image=None,
                )

                samples.append(sample)

        return samples

    def load_raw(self, number=500) -> List[Sample]:
        samples = []
        
        folder = Path(self.folder_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        image_paths = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in image_extensions
        )
        
        for path in image_paths[:number]:
            samples.append(Sample(
                image_path=str(path),
                lanes=[],
                h_samples=[float(x) for x in TUSIMPLE_H_SAMPLES],
                image=None,
            ))
        
        return samples
    
    def load_image(self, sample: Sample) -> Sample:
        image = cv2.imread(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at '{sample.image_path}'")
        sample.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return sample
    
    def load_images(self, samples: List[Sample]) -> List[Sample]:
        return [self.load_image(sample) for sample in samples]