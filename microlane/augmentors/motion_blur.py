import numpy as np
import cv2

from microlane.schemas.sample import Sample

def motion_blur(sample: Sample, value: float) -> Sample:

    # value: 0.0 = no blur, 0.1 = maximum motion blur
    # scale value to a kernel size
    max_kernel = 21
    kernel_size = max(1, int(value * max_kernel * 10))
    kernel_size = kernel_size + (1 - kernel_size % 2)  # ensure odd

    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    sample.image = cv2.filter2D(sample.image, -1, kernel)
    sample.motion_blur = value

    return sample
