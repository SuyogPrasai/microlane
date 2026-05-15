import numpy as np

from microlane.schemas.sample import Sample

def lighting(sample: Sample, value: float) -> Sample:

    # value: -1.0 = darkest, 0.0 = no change, 1.0 = brightest
    sample.image = np.clip(sample.image.astype(np.int32) + int(value * 255), 0, 255).astype(np.uint8)
    sample.lighting = value

    return sample