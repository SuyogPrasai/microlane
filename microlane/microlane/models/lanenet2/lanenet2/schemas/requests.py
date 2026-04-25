from typing import List, Optional
import numpy as np
from pydantic import BaseModel, validator
from microlane.models.lanenet2.lanenet2.schemas.api_schemas import Sample # type: ignore


class LaneLineRequest(BaseModel):
    x_coordinates: List[float]
    y_coordinates: List[float]

    @validator("y_coordinates")
    def x_and_y_length_match(cls, value, values):
        x_coordinates = values.get("x_coordinates")
        if x_coordinates is not None and len(value) != len(x_coordinates):
            raise ValueError("x_coordinates and y_coordinates must have the same length")
        return value


class SampleRequest(BaseModel):
    image_path: str
    image: List
    lanes: List
    h_samples: List
      
    brightness: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0
    blur: float = 0.0


    def to_sample(self):
        return Sample(
            image_path=self.image_path,
            image=np.array(self.image, dtype="uint8"),
            lanes=np.array(self.lanes),
            h_samples=np.array(self.h_samples),
            brightness=self.brightness,
            blur=self.blur,
            zoom=self.zoom,
            rotation=self.rotation
        )

class InferRequest(BaseModel):
    sample: SampleRequest


class BatchInferRequest(BaseModel):
    samples: List[SampleRequest]