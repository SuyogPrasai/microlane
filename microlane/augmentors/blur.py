import cv2

from microlane.schemas.sample import Sample

def blur(sample: Sample, value: float) -> Sample:

    # value: 0.0 = no blur, 1.0 = maximum blur
    # kernel size must be odd, so we scale value to a range of odd integers
    max_kernel = 21
    kernel_size = int(value * max_kernel)
    kernel_size = max(1, kernel_size + (1 - kernel_size % 2))  # ensure odd

    sample.image = cv2.GaussianBlur(sample.image, (kernel_size, kernel_size), 0)
    sample.blur = value

    return sample
