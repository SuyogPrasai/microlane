import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torchvision.transforms as transforms  # pyright: ignore[reportMissingImports]

from PIL import Image

from schemas.api_schemas import Sample
from engine import UFLDEngine


class PreProcessor:

    def __init__(
        self,
        engine: UFLDEngine,
    ) -> None:

        self.engine = engine

        self._img_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.engine._NET_H,
                        self.engine._NET_W,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def process(
        self,
        sample: Sample,
    ) -> torch.Tensor:

        image = sample.image

        image = np.squeeze(image)

        if image.dtype != np.uint8:

            if image.max() <= 1.0:
                image = (
                    image * 255.0
                ).clip(0, 255)

            image = image.astype(np.uint8)

        rgb = image[:, :, ::-1].copy()

        pil_img = Image.fromarray(rgb)

        tensor: torch.Tensor = self._img_transforms(
            pil_img
        )

        return tensor.unsqueeze(0).to(
            self.engine.device
        )