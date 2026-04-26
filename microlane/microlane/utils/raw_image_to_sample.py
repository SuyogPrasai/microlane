from microlane.schema.sample import Sample
import numpy as np
import cv2

def imageToSample(image_path: str) -> Sample:
    
    image = cv2.imread(image_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
   
    return Sample(
        image=image,
        image_path=image_path,
        lanes=[],
        h_samples=[]
    )
     