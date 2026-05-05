from pydantic import BaseModel, validator # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from typing import List, Optional # pyright: ignore[reportMissingImports]

from schemas.api_schemas import Sample # pyright: ignore[reportMissingImports]

class SampleRequest(BaseModel):
    image_path: str
    image: List
    lanes: List
    h_samples: List
    dataset: str
    
    blur: float = 0.0
    lighting: float = 1.0
    rotation: float = 0.0
    zoom: float = 1.0
    motion_blur: float = 0.0
    
class InferRequest(BaseModel):
    sample: SampleRequest