import time
import numpy as np
import torch # pyright: ignore[reportMissingImports]
import scipy.special # pyright: ignore[reportMissingImports]
from PIL import Image
from typing import Tuple

from model.model import parsingNet # type: ignore
from data.constant import culane_row_anchor, tusimple_row_anchor  # type: ignore

class UFLDEngine():
    
    def __init__(self,
                 weights_path: str,
                backbone: str = "18",
                griding_num: int = 100,
                use_gpu: bool = True,
                ) -> None:
        
        self.weights_path = weights_path
        
        self._NET_H, self._NET_W = [288, 800]
        
        self.dataset = "tusimple"
        
        self.backbone = backbone
        
        self.griding_num = griding_num
        
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        
        self.cls_num_per_lane = 56

        self.row_anchor = tusimple_row_anchor
        
        
        col_sample = np.linspace(0, self._NET_W - 1, self.griding_num)
        
        self.col_sample_w: float = col_sample[1] - col_sample[0]

        self.net: parsingNet | None = None
        
        self._build_graph()
        
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
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
        
        t_start = time.time()
        
        with torch.no_grad():
            
            raw_out = self.net(image)
            
        t_cost = time.time() - t_start
        
        return raw_out, t_cost
        
        
    def _build_graph(self):
        
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
