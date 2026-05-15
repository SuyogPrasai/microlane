from typing import List

import numpy as np
from PIL import Image
from torchvision import transforms # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]

from schemas.api_schemas import Sample

import config # pyright: ignore[reportMissingImports]

IMG_H = config.img_height
IMG_W = config.img_width


TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

class PreProcessor():
    
    def __init__(self) -> None:
        pass
    
    def process(self, samples: List[Sample]) -> List[torch.Tensor]:
        
        frames = [s.image for s in samples]

        while len(frames) < 5:
            frames.insert(0, frames[0])
    
        frames = frames[-5:]

        tensors = [self._preprocess(f) for f in frames]

        return tensors
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        # Convert Image from HxWx3 uint8 numpy array RGB to (C,H,W) Tensor

        pil = Image.fromarray(image)

        return TRANSFORM(pil) # pyright: ignore[reportReturnType]