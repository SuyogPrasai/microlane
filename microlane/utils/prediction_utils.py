# microlane/utils/prediction_io.py

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from typing import Union

from microlane.schemas.prediction import Prediction

class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy arrays and scalars to JSON-serialisable Python types."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


def _sample_to_dict(sample) -> dict:
    return {
        "image_path": sample.image_path,
        "lanes":      sample.lanes,       # handled by encoder
        "h_samples":  sample.h_samples,
        "dataset":    sample.dataset,
        "blur":       sample.blur,
        "lighting":   sample.lighting,
        "rotation":   sample.rotation,
        "zoom":       sample.zoom,
        "motion_blur": sample.motion_blur,
        # image (np.ndarray) is intentionally omitted — too large for JSON
    }


def store_prediction(prediction: Prediction, path: Union[str, Path, None] = None) -> Path:
    """
    Serialise a Prediction object to a JSON file.

    Numpy arrays are converted to nested lists.
    The raw image pixels inside each Sample are skipped (image_path is kept).

    Parameters
    ----------
    prediction : Prediction
        The prediction object to persist.
    path : str | Path | None
        Destination file path.  If None, a timestamped file is created in the
        current working directory: ``prediction_<YYYYMMDD_HHMMSS>.json``.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"prediction_{timestamp}.json")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_time": prediction.run_time,
        "lanes":    prediction.lanes,      # encoder handles ndarray → list
        "h_samples": prediction.h_samples,
        "samples":  [_sample_to_dict(s) for s in prediction.samples],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, cls=_NumpyEncoder, indent=2)

    return path.resolve()


# ── Visualisation ─────────────────────────────────────────────────────────────

# Colours for up to 6 lanes; extend if needed
_PRED_COLOURS = ["#00FF7F", "#00CFFF", "#FFD700", "#FF69B4", "#FF8C00", "#BF5FFF"]
_GT_COLOURS   = ["#006400", "#005F8A", "#8B6914", "#8B0040", "#8B4500", "#5A0080"]


def visualize_prediction(prediction: Prediction, sample_index: int = 0) -> None:
    """
    Overlay predicted and ground-truth lane lines on a single sample image.

    Parameters
    ----------
    prediction : Prediction
        The prediction object (must contain at least one Sample).
    sample_index : int
        Which sample to display (0-based).  Iterate over this to step through
        all samples:

            for i in range(len(prediction.samples)):
                visualize_prediction(prediction, sample_index=i)
                plt.show()

    Notes
    -----
    * Lanes are drawn as scatter dots at (x, y) pairs where x != -2
      (the TuSimple sentinel for missing points).
    * Predicted lanes come from ``prediction.lanes`` /
      ``prediction.h_samples``.
    * Ground-truth lanes come from the corresponding ``Sample.lanes`` /
      ``Sample.h_samples``.
    """
    n = len(prediction.samples)
    if not (0 <= sample_index < n):
        raise IndexError(f"sample_index {sample_index} is out of range for {n} sample(s).")

    sample = prediction.samples[sample_index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Sample {sample_index + 1}/{n}  |  "
        f"dataset: {sample.dataset}  |  "
        f"run_time: {prediction.run_time:.3f}s",
        fontsize=12, fontweight="bold",
    )

    h_samples_pred = np.asarray(prediction.h_samples)
    h_samples_gt   = np.asarray(sample.h_samples)

    for ax, lanes_arr, h_samples, colours, title in [
        (axes[0], prediction.lanes, h_samples_pred, _PRED_COLOURS, "Predicted lanes"),
        (axes[1], sample.lanes,     h_samples_gt,   _GT_COLOURS,   "Ground truth lanes"),
    ]:
        ax.imshow(sample.image, origin="upper")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

        lanes = np.asarray(lanes_arr)          # shape: (n_lanes, n_h_samples)
        legend_handles = []

        for lane_idx, lane_xs in enumerate(lanes):
            colour = colours[lane_idx % len(colours)]
            lane_xs = np.asarray(lane_xs)

            # Filter out TuSimple -2 sentinel (missing points)
            valid = lane_xs != -2
            xs = lane_xs[valid]
            ys = h_samples[valid]

            if xs.size == 0:
                continue

            ax.scatter(xs, ys, s=6, color=colour, linewidths=0)
            ax.plot(xs, ys, color=colour, linewidth=1.5, alpha=0.7)

            legend_handles.append(
                mpatches.Patch(color=colour, label=f"Lane {lane_idx + 1}")
            )

        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right",
                      fontsize=8, framealpha=0.6)

    plt.tight_layout()