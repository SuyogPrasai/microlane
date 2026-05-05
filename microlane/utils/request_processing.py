import numpy as np 

from microlane.schemas.sample import Sample
from microlane.schemas.prediction import Prediction

def sample_to_payload(sample: Sample) -> dict:
    return {
        "image_path": sample.image_path,
        "image": sample.image.tolist(),
        "lanes": sample.lanes.tolist(),
        "h_samples": sample.h_samples.tolist(),
        "dataset": sample.dataset,
        
        "blur": sample.blur,
        "lighting": sample.lighting,
        "zoom": sample.zoom,
        "rotation": sample.rotation,
        "motion_blur": sample.motion_blur
    }
    
    
def payload_to_prediction(payload: dict) -> Prediction:
    return Prediction(
        samples=[
            Sample(
                image_path=s["image_path"],
                image=np.array(s["image"]),
                lanes=np.array(s["lanes"]),
                h_samples=np.array(s["h_samples"]),
                dataset=s["dataset"],
                
                blur=s["blur"],
                lighting=s["lighting"],
                zoom=s["zoom"],
                rotation=s["rotation"],
                motion_blur=s["motion_blur"]
            )
            for s in payload["samples"]
        ],
        
        lanes=np.array(payload["lanes"]),
        h_samples=np.array(payload["h_samples"]),
        run_time=payload["run_time"]
    )