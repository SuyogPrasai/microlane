import cv2
import numpy as np

from microlane.schema.sample import Sample
from microlane.augmentors.augmentor import Augmentor


def load_image_from_sample(sample: Sample) -> Sample:

    if sample.image is not None:
        return sample

    sample.image = cv2.imread(sample.image_path)

    auggie = Augmentor()

    sample = auggie.blur(sample)

    sample = auggie.zoom(sample)

    sample = auggie.rotation(sample)
    
    sample = auggie.brightness(sample)

    return sample