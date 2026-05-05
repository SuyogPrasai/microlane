from typing import Tuple

from microlane.schemas.prediction import Prediction
from microlane.evalutation.lane_eval import LaneEval


def calculate_f1_score(prediction: Prediction) -> Tuple[float, float, float]:
    """
    Calculate F1 score, precision, and recall for a prediction against
    the last sample's ground truth lanes.

    Args:
        prediction: A Prediction object containing samples (with ground truth
                    lanes), predicted lanes (x values), h_samples (y values),
                    and run_time in milliseconds.

    Returns:
        Tuple of (f1_score, precision, recall), each in [0.0, 1.0].
    """
    if not prediction.samples:
        return 0.0, 0.0, 0.0

    sample = prediction.samples[-1]

    gt_lanes: list[list[float]] = [lane.tolist() for lane in sample.lanes]
    pred_lanes: list[list[float]] = [lane.tolist() for lane in prediction.lanes]
    y_samples: list[float] = prediction.h_samples.tolist()

    if not gt_lanes or not pred_lanes:
        return 0.0, 0.0, 0.0

    try:
        _accuracy, fp, fn = LaneEval.bench(
            pred=pred_lanes,
            gt=gt_lanes,
            y_samples=y_samples,
            running_time=prediction.run_time,
        )

        precision = 1.0 - fp
        recall = 1.0 - fn
        denom = precision + recall
        f1_score = (2 * precision * recall / denom) if denom > 0 else 0.0

        return f1_score, precision, recall
    except Exception:
        return 0.0, 0.0, 0.0