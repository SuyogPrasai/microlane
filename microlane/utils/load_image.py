import cv2
import numpy as np

from microlane.schema.sample import Sample
from microlane.schema.output import ModelPrediction
from microlane.augmentors.augmentor import Augmentor

def load_image_from_sample(sample: Sample) -> Sample:

    if sample.image is not None:
        return sample

    sample.image = cv2.imread(sample.image_path)

    # auggie = Augmentor()

    # sample = auggie.blur(sample)

    # sample = auggie.zoom(sample)

    # sample = auggie.rotation(sample)
    
    # sample = auggie.brightness(sample)

    return sample

def parse_prediction(response) -> ModelPrediction:
    prediction = response.json()
    
    return ModelPrediction(
        sample=Sample(
            image=prediction['sample']['image'],
            image_path=prediction['sample']['image_path'],
            h_samples=prediction['sample']['h_samples'],
            lanes=prediction['sample']['lanes'],
            blur=prediction['sample']['blur'],
            brightness=prediction['sample']['brightness'],
            zoom=prediction['sample']['zoom'],
            rotation=prediction['sample']['rotation']
        ),
        lanes=prediction['lanes'],
        run_time=prediction['run_time']
    )