from typing import List, Optional
import numpy as np
from pydantic import BaseModel, validator

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
    image: Optional[List]
    actual_lanes: List[LaneLineRequest]

    def to_sample(self):
        from schemas.api_schemas import LaneLine, Sample # type: ignore
        return Sample(
            image_path=self.image_path,
            image=np.array(self.image, dtype="uint8") if self.image is not None else None,
            actual_lanes=[
                LaneLine(x_coordinates=lane.x_coordinates, y_coordinates=lane.y_coordinates)
                for lane in self.actual_lanes
            ],
        )

class InferRequest(BaseModel):
    sample: SampleRequest


class BatchInferRequest(BaseModel):
    samples: List[SampleRequest]