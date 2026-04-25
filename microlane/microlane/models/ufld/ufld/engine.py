import time
import numpy as np
import torch
import torchvision.transforms as transforms
import scipy.special
from PIL import Image
from typing import Tuple

from model.model import parsingNet # type: ignore
from data.constant import culane_row_anchor, tusimple_row_anchor  # type: ignore

class UFLDEngine():
    
    # Canonical input resolution expected by parsingNet
    _NET_H = 288
    _NET_W = 800
    
    def __init__(
        self, 
        weights_path,
        dataset: str = "tusimple",
        backbone: str = "18",
        griding_num: int = 100,
        use_gpu: bool = True,
        ) -> None:
        
        self.weights_path = weights_path
        
        self.backbone = backbone
        
        self.griding_num = griding_num
        
        self.dataset = dataset
        
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        
        if self.dataset == "tusimple":
            
            self.cls_num_per_lane = 56

            self.row_anchor = tusimple_row_anchor

        elif self.dataset == "culane":
            
            self.cls_num_per_lane = 18

            self.row_anchor = culane_row_anchor

        else:
            raise ValueError(f"Unknown dataset: {dataset!r}. Use 'tusimple' or 'culane'.")
        
        col_sample = np.linspace(0, self._NET_W - 1, self.griding_num)
        
        self.col_sample_w: float = col_sample[1] - col_sample[0]

        self.net: parsingNet | None = None
        
        self._img_transforms: transforms.Compose | None = None

        self._build_graph()

        
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run a single forward pass.

        Args:
            image: uint8 BGR numpy array (H, W, 3) — the *original* image
                   at whatever resolution it was loaded; preprocessing is
                   handled internally.

        Returns:
            out_j:  float32 array of shape (cls_num_per_lane, num_lanes)
                    containing the predicted x-grid indices (0 = no lane).
            t_cost: wall-clock seconds for the forward pass only.
        """
        
        if self.net is None:
            raise RuntimeError("Model not loaded. Call _build_graph() first.")
    
        input_tensor = self.image_to_tensor(image)

        t_start = time.time()
        
        with torch.no_grad():
            
            raw_out = self.net(input_tensor)
            
        t_cost = time.time() - t_start
        
        out_j = self._postprocess(raw_out)
        
        return out_j, t_cost

    
    def close(self) -> None:
        """Release GPU memory."""
        
        if self.net is not None:
            del self.net
            self.net = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    
    def _build_graph(self) -> None:

        """Load parsingNet weights and build the transform pipeline."""

        self.net = parsingNet(
            pretrained=False,
            backbone=str(self.backbone) if hasattr(self, "backbone") else "18",
            cls_dim=(self.griding_num + 1, self.cls_num_per_lane, 4),
            use_aux=False,
        ).to(self.device)
        
        state_dict = torch.load(self.weights_path, map_location="cpu")["model"]
        
        # Strip the 'module.' prefix added by DataParallel, if present
        compatible_state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    
        self.net.load_state_dict(compatible_state_dict, strict=False) # type: ignore
        
        self.net.eval() # type: ignore

        self._img_transforms = transforms.Compose(
            [
                transforms.Resize((self._NET_H, self._NET_W)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    
    def image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        
        image = np.squeeze(image)
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255)
            image = image.astype(np.uint8)

        rgb = image[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb)
        
        assert self._img_transforms is not None, "_build_graph() was not called"
        
        tensor: torch.Tensor = torch.Tensor(self._img_transforms(pil_img))
        
        return tensor.unsqueeze(0).to(self.device)
    
    def _postprocess(self, raw_out: torch.Tensor) -> np.ndarray:
        
        out_j = raw_out[0].data.cpu().numpy()
        
        out_j = out_j[:, ::-1, :]  # flip row order: bottom → top
        
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        
        idx = (np.arange(self.griding_num) + 1).reshape(-1, 1, 1)

        loc = np.sum(prob * idx, axis=0)
        
        # Where argmax points to the 'no-lane' bin, zero out the location
        argmax_j = np.argmax(out_j, axis=0)
        
        loc[argmax_j == self.griding_num] = 0

        return loc  # (cls_num_per_lane, num_lanes)