
from microlane.schema.sample import Sample
import cv2, numpy as np

def load_image_from_sample(sample: Sample) -> Sample:
    
    if sample.image is not None:
        
        return sample
    
    image = cv2.imread(sample.image_path)
    
    sample.image = image
    
    return sample