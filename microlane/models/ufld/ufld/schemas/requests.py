from typing import List, Optional
import numpy as np
from pydantic import BaseModel, validator
from schemas.api_schemas import Sample # type: ignore


class SampleRequest(BaseModel):
    image_path: str
    image: List
    lanes: List
    h_samples: List
      
    lighting: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0
    blur: float = 0.0


    def to_sample(self):
        return Sample(
            image_path=self.image_path,
            image=np.array(self.image, dtype="uint8"),
            lanes=np.array(self.lanes),
            h_samples=np.array(self.h_samples),
            lighting=self.lighting,
            blur=self.blur,
            zoom=self.zoom,
            rotation=self.rotation
        )

class InferRequest(BaseModel):
    sample: SampleRequest


class BatchInferRequest(BaseModel):
    samples: List[SampleRequest]