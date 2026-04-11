from typing import Tuple, List
from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
import numpy as np
import cv2


class LaneNet2Formatter:
    
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size
        
    def format_one(self, sample: Sample) -> LaneNet2Input:
        if sample.image is None:
            raise ValueError(
                f"Sample '{sample.image_path}' has no loaded image. "
                "Call sample.load() before formatting."
            )
 
        W, H = self.target_size
        image: np.ndarray = sample.image
        orig_h, orig_w = image.shape[:2]
 
        resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 127.5 - 1.0
        
        binary_mask   = np.zeros((orig_h, orig_w), dtype=np.uint8)
        instance_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
 
        for lane_id, lane in enumerate(sample.lanes, start=1):
            for x, y in zip(lane.x_coordinates, lane.y_coordinates):
                if x < 0:          # -2 sentinel → no lane at this row
                    continue
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < orig_h and 0 <= xi < orig_w:
                    binary_mask[yi, xi]   = 1
                    instance_mask[yi, xi] = lane_id
 
        binary_mask   = cv2.resize(binary_mask,   (W, H), interpolation=cv2.INTER_NEAREST)
        instance_mask = cv2.resize(instance_mask, (W, H), interpolation=cv2.INTER_NEAREST)
 
        return LaneNet2Input(
            image=resized,
            binary_mask=binary_mask,
            instance_mask=instance_mask,
        )
    
    def batch_format(self, samples: List[Sample]) -> List[LaneNet2Input]:
        return [self.format_one(sample) for sample in samples]