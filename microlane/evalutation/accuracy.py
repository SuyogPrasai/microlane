from typing import List

from microlane.schemas.prediction import Prediction, Evaluation
from microlane.evalutation.lane_eval import LaneEval

def calculate_accuracy(prediction: Prediction) -> float:
    
    if not prediction.samples:
        return 0.0
    
    sample = prediction.samples[-1]
    
    gt_lanes: list[list[float]] = [lane.tolist() for lane in sample.lanes]
    pred_lanes: list[list[float]] = [lane.tolist() for lane in prediction.lanes]
    y_samples: list[float] = prediction.h_samples.tolist()
    
    if not gt_lanes or not pred_lanes:
        return 0.0
    
    try:
        accuracy, _fp, _fn = LaneEval.bench(
            pred=pred_lanes,
            gt=gt_lanes,
            y_samples=y_samples,
            running_time=prediction.run_time,
        )
        return float(accuracy)

    except Exception:
        return 0.0