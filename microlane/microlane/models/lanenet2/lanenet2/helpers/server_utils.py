import numpy as np
from microlane.models.lanenet2.lanenet2.schemas.api_schemas import ModelPrediction
from microlane.models.lanenet2.lanenet2.schemas.requests import SampleRequest


def prediction_to_dict(pred: ModelPrediction) -> dict:
    return {
        "sample": {
            "image_path": pred.sample.image_path,
            "image": None if pred.sample.image is None else pred.sample.image.tolist(),
            "lanes": np.array(pred.sample.lanes).tolist(),
            "h_samples": np.array(pred.sample.h_samples).tolist(),
            "blur": pred.sample.blur,
            "zoom": pred.sample.zoom,
            "rotation": pred.sample.rotation,
            "lighting": pred.sample.lighting,
        },
        
        "lanes": np.array(pred.lanes).tolist(),
        "run_time": pred.run_time 
    }


def sample_request_to_dict(sample: SampleRequest) -> dict:
    return {
        "image": None if sample.image is None else np.array(sample.image).tolist(),
        "image_path": sample.image_path,
        "lanes": sample.lanes
    }