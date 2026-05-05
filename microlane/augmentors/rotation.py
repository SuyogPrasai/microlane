import numpy as np
import cv2

from microlane.schemas.sample import Sample

def rotation(sample: Sample, value: float) -> Sample:

    # value: 0.0 - 360.0 degrees
    h, w = sample.image.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), value, 1.0)

    # rotate image
    sample.image = cv2.warpAffine(sample.image, M, (w, h))

    # rotate lane points
    lanes = np.asarray(sample.lanes, dtype=np.float32)       # (n_lanes, n_h_samples)
    h_samples = np.asarray(sample.h_samples, dtype=np.float32)

    for lane_idx, lane_xs in enumerate(lanes):

        for pt_idx, x in enumerate(lane_xs):

            if x == -2:
                continue

            y = h_samples[pt_idx]

            # apply rotation matrix
            rotated = M @ np.array([x, y, 1.0])

            # clip to image bounds
            lanes[lane_idx, pt_idx] = np.clip(rotated[0], 0, w - 1)

    sample.lanes = lanes
    sample.rotation = value

    return sample