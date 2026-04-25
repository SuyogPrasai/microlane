import cv2
import numpy as np
from microlane.schema.sample import Sample


class Augmentor():

    def __init__(self) -> None:
        pass

    def blur(self, sample: Sample) -> Sample:
        if sample.image is None or sample.blur <= 0.0:
            return sample

        image: np.ndarray = sample.image
        blur_value = sample.blur

        ksize = max(3, int(blur_value * 20))
        if ksize % 2 == 0:
            ksize += 1

        sample.image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return sample

    def zoom(self, sample: Sample) -> Sample:
        if sample.image is None or sample.zoom == 1.0:
            return sample

        image: np.ndarray = sample.image
        zoom_value = sample.zoom

        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_value), int(w * zoom_value)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if zoom_value > 1.0:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            sample.image = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            canvas = np.zeros_like(image)
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            sample.image = canvas

        return sample

    def rotation(self, sample: Sample) -> Sample:
        if sample.image is None or sample.rotation == 0.0:
            return sample

        image: np.ndarray = sample.image
        rotation_value = sample.rotation

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, rotation_value, scale=1.0)
        sample.image = cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return sample

    def brightness(self, sample: Sample) -> Sample:
        if sample.image is None or sample.brightness == 1.0:
            return sample

        image: np.ndarray = sample.image

        sample.image = cv2.convertScaleAbs(image, alpha=sample.brightness, beta=0)

        return sample
    

    def motion_blur(self, sample: Sample) -> Sample:
        if sample.image is None or sample.motion_blur <= 0.0:
            return sample

        image: np.ndarray = sample.image
        motion_blur_value = sample.motion_blur  # 0.0 – 1.0

        # Kernel size controls how strong the shake blur is
        kernel_size = max(3, int(motion_blur_value * 30))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Random angle simulates unpredictable shake direction
        angle = np.random.uniform(0, 360)
        angle_rad = np.deg2rad(angle)

        # Build motion blur kernel along the random angle
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            offset = i - center
            x = center + int(round(offset * np.cos(angle_rad)))
            y = center + int(round(offset * np.sin(angle_rad)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel /= kernel.sum()

        sample.image = cv2.filter2D(image, -1, kernel)

        return sample