import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from microlane.schemas.prediction import Prediction
from microlane.schemas.sample import Sample



class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


_PRED_COLOURS = ["#00FF7F", "#00CFFF", "#FFD700", "#FF69B4", "#FF8C00", "#BF5FFF"]
_GT_COLOURS   = ["#006400", "#005F8A", "#8B6914", "#8B0040", "#8B4500", "#5A0080"]


@dataclass
class Experiment:

    base_dir: Path
    folder: Path = field(init=False)
    prediction_path: Optional[Path] = field(init=False, default=None)
    

    def __post_init__(self):

        self.base_dir = Path(self.base_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

        with self.prediction_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, cls=_NumpyEncoder, indent=2)

        return self.prediction_path.resolve()
    
    def visualize_prediction(self, prediction: Prediction, show: bool = False) -> None:

        sample: Sample = prediction.samples[-1]

        h_pred = np.asarray(prediction.h_samples)
        h_gt   = np.asarray(sample.h_samples)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"dataset: {sample.dataset}  |  run_time: {prediction.run_time:.3f}s",
            fontsize=12, fontweight="bold",
        )

        panels = [
            (axes[0], prediction.lanes, h_pred, _PRED_COLOURS, "Predicted lanes"),
            (axes[1], sample.lanes,     h_gt,   _GT_COLOURS,   "Ground truth lanes"),
        ]

        h, w = sample.image.shape[:2]

        for ax, lanes_arr, h_samples, colours, title in panels:

            ax.imshow(sample.image, origin="upper")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

            legend_handles = []

            for lane_idx, lane_xs in enumerate(np.asarray(lanes_arr)):

                colour  = colours[lane_idx % len(colours)]
                lane_xs = np.asarray(lane_xs)
                valid   = lane_xs != -2
                xs, ys  = lane_xs[valid], h_samples[valid]

                if xs.size == 0:
                    continue

                ax.scatter(xs, ys, s=6, color=colour, linewidths=0)
                ax.plot(xs, ys, color=colour, linewidth=1.5, alpha=0.7)
                legend_handles.append(mpatches.Patch(color=colour, label=f"Lane {lane_idx + 1}"))

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.6)

        plt.tight_layout()
        out_path = self.folder / "inference" / f"visualization_{self.visualization_count:04d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        self.visualization_count += 1

        if show:
            plt.show()