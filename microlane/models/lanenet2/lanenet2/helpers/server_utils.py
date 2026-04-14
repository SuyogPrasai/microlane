import numpy as np
from schemas.api_schemas import LaneNet2Output
from schemas.requests import SampleRequest


def prediction_to_dict(pred: LaneNet2Output) -> dict:
    return {
        "binary_segmentation": pred.binary_segmentation.tolist()
            if isinstance(pred.binary_segmentation, np.ndarray)
            else pred.binary_segmentation,
        "instance_segmentation": pred.instance_segmentation.tolist()
            if isinstance(pred.instance_segmentation, np.ndarray)
            else pred.instance_segmentation,
    }


def sample_request_to_dict(sample: SampleRequest) -> dict:
    return {
        "image_path": sample.image_path,
        "image": None if sample.image is None else np.asarray(sample.image).tolist(),
        "actual_lanes": [
            {"x_coordinates": lane.x_coordinates, "y_coordinates": lane.y_coordinates}
            for lane in sample.actual_lanes
        ],
    }