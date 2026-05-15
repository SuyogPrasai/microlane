import os, sys, time
import numpy as np

import torch # pyright: ignore[reportMissingImports]

REPO_DIR = os.path.join(os.path.dirname(__file__), "LaneDetectionCode")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from model import generate_model # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

LSTM_MODELS = {"UNet-ConvLSTM", "SegNet-ConvLSTM"}

class _Args:

    def __init__(self, model_name: str, use_cuda: bool = True):
        self.model = model_name
        self.cuda = use_cuda
        self.seed = 1

class RLDEngine():
    
    def __init__(self,
                 model_name: str,
                 weights_path: str,
                 use_gpu: bool = True) -> None:
        
        self.model_name = model_name
        
        self.device = torch.device(
            "cuda"
            if use_gpu and torch.cuda.is_available()
            else "cpu"
        )
        
        self.model = self._load_model(weights_path)
        
    
    def predict(self, tensors: list[torch.Tensor]) -> tuple[np.ndarray, float]:
        
        t_start = time.time()
        
        with torch.no_grad():
            
            inp = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)

            result: torch.Tensor | tuple[torch.Tensor, ...] = self.model(inp)

            output: torch.Tensor = result[0] if isinstance(result, tuple) else result

            mask: torch.Tensor = output.argmax(dim=1).squeeze(0)
        
        t_cost: float = time.time() - t_start
        
        return mask.cpu().numpy().astype(np.uint8), t_cost
    

    def _load_model(self, weights_path: str) -> torch.nn.Module:
        
        args = _Args(
            self.model_name,
            use_cuda=(self.device.type == "cuda"),
        )
        
        model = generate_model(args)

        pretrained_dict = torch.load(
            weights_path,
            map_location=self.device,
        )
        
        model_dict = model.state_dict()

        filtered = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict
        }

        model_dict.update(filtered)

        model.load_state_dict(model_dict)

        print(
            f"[Engine] loaded "
            f"{len(filtered)}/{len(pretrained_dict)} "
            f"keys from {weights_path}"
        )

        return model.to(self.device).eval()