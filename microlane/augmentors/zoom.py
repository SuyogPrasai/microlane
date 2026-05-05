import cv2
import numpy as np

from microlane.schemas.sample import Sample

def zoom(sample: Sample, value: float) -> Sample:

    # value: 1.0 = no zoom, 3.0 = maximum zoom
    h, w = sample.image.shape[:2]
    cx, cy = w / 2, h / 2

    # crop region shrinks as zoom increases
    crop_w = int(w / value)
    crop_h = int(h / value)

    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    # crop then resize back to original dimensions
    cropped = sample.image[y1:y2, x1:x2]
    sample.image = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # scale lane points to match resized image
    lanes = np.asarray(sample.lanes, dtype=np.float32)
    h_samples = np.asarray(sample.h_samples, dtype=np.float32)

    scale_x = w / crop_w
    scale_y = h / crop_h

    for lane_idx, lane_xs in enumerate(lanes):

        for pt_idx, x in enumerate(lane_xs):

            if x == -2:
                continue

            y = h_samples[pt_idx]

            scaled_x = (x - x1) * scale_x
            scaled_y = (y - y1) * scale_y

            # clip to image bounds
            lanes[lane_idx, pt_idx] = np.clip(scaled_x, 0, w - 1)
            h_samples[pt_idx] = np.clip(scaled_y, 0, h - 1)

    sample.lanes = lanes
    sample.h_samples = h_samples
    sample.zoom = value

    return sample