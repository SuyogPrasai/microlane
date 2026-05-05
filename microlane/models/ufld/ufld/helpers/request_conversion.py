from schemas.api_schemas import Sample, Prediction # pyright: ignore[reportMissingImports]
from schemas.requests import SampleRequest # pyright: ignore[reportMissingImports]

import numpy as np # pyright: ignore[reportMissingImports]

def sample_request_to_sample(sample_request: SampleRequest) -> Sample:
    return Sample(
        image_path=sample_request.image_path,
        image=np.array(sample_request.image, dtype=np.uint8),
        lanes=np.array(sample_request.lanes),
        h_samples=np.array(sample_request.h_samples),
        dataset=sample_request.dataset,
        
        blur=sample_request.blur,
        lighting=sample_request.lighting,
        rotation=sample_request.rotation,
        zoom=sample_request.zoom,
        motion_blur=sample_request.motion_blur
    )
    
def prediction_to_response(pred: Prediction) -> dict:
    
    samples = pred.samples
    
    return {
        "samples": [
            {
                "image_path": sample.image_path,
                "image": sample.image.tolist(),
                "lanes": np.array(sample.lanes).tolist(),
                "h_samples": np.array(sample.h_samples).tolist(),
                "dataset":    sample.dataset,
                "blur": sample.blur,
                "zoom": sample.zoom,
                "rotation": sample.rotation,
                "lighting": sample.lighting,
                "motion_blur": sample.motion_blur
            }
            for sample in samples
        ],
        
        "lanes": np.array(pred.lanes).tolist(),        
        "h_samples": np.array(pred.h_samples).tolist(),
        "run_time": pred.run_time,
    }