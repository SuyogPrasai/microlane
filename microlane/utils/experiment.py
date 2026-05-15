
from zoneinfo import ZoneInfo
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from microlane.schemas.prediction import Prediction

_PRED_COLOURS = ["#00FF7F", "#00CFFF", "#FFD700", "#FF69B4", "#FF8C00", "#BF5FFF"]
_GT_COLOURS   = ["#006400", "#005F8A", "#8B6914", "#8B0040", "#8B4500", "#5A0080"]

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)

class Experiment():
    
    def __init__(self, base_dir) -> None:

        self.base_dir = Path(base_dir)

        timestamp = datetime.now(tz=ZoneInfo("Asia/Kathmandu")).strftime("%Y-%m-%d_%H-%M-%S")
        
        self.folder = self.base_dir / f"experiment_{timestamp}"
        
        self.folder.mkdir(parents=True, exist_ok=True)
        
        self.visualization_count = 0
        
    def store_prediction(self, prediction: Prediction) -> Path:
        
        payload = {
            "run_time":  prediction.run_time,
            "lanes":     prediction.lanes,
            "h_samples": prediction.h_samples,
            "samples": [
                {
                    "image_path":  s.image_path,
                    "lanes":       s.lanes,
                    "h_samples":   s.h_samples,
                    "dataset":     s.dataset,
                    "blur":        s.blur,
                    "lighting":    s.lighting,
                    "rotation":    s.rotation,
                    "zoom":        s.zoom,
                    "motion_blur": s.motion_blur,
                }
                for s in prediction.samples
            ],
        }
        
        self.prediction_path = self.folder / "prediction.json"

        existing = []
        
        if self.prediction_path.exists():
            with self.prediction_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
                
        existing.append(payload)

        with self.prediction_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, cls=_NumpyEncoder)

        return self.prediction_path.resolve()
    
    def visualize_prediction(self, prediction: Prediction, show: bool = False):
        
        sample = prediction.samples[-1]

        fig, ax = plt.subplots(figsize=(12, 6))

        title = Path(sample.image_path).parts
        
        ax.set_title("/".join(title[-3:]), fontsize=8)
          
        img = sample.image
        
        ax.imshow(img)

        ax.axis("off")
        
        for lane_idx, lane in enumerate(sample.lanes):
        
            colour = _GT_COLOURS[lane_idx % len(_GT_COLOURS)]
            
            valid = [(x, y) for x, y in zip(lane, sample.h_samples) if x != -2]
            
            if valid:

                xs, ys = zip(*valid)
                
                ax.plot(xs, ys, color=colour, linewidth=2, linestyle="--", label=f"GT Lane {lane_idx}")
    
        
        for lane_idx, lane in enumerate(prediction.lanes):
        
            colour = _PRED_COLOURS[lane_idx % len(_PRED_COLOURS)]
            
            valid = [(x, y) for x, y in zip(lane, prediction.h_samples) if x != -2]
            
            if valid:
                xs, ys = zip(*valid)
                
                ax.plot(xs, ys, color=colour, linewidth=2, label=f"Pred Lane {lane_idx}")
                
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
        
        out_folder = self.folder / "inference"
        
        out_folder.mkdir(parents=True, exist_ok=True)
        
        out_path = out_folder  / f"viz_{self.visualization_count:04d}.png"
        
        fig.savefig(out_path, bbox_inches="tight", dpi=150)

        self.visualization_count += 1
        
        if show:
            plt.show()

        plt.close(fig)
