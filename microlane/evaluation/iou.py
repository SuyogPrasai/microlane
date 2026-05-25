from microlane.schemas.prediction import Prediction
from microlane.evaluation.core.lane_iou import LaneIoU


def calculate_iou(prediction: Prediction) -> float:
    
    gt_sample  = prediction.samples[-1]
    
    gt_lanes   = gt_sample.lanes
    
    pred_lanes = prediction.lanes
    
    y_samples  = prediction.h_samples
    
    return LaneIoU.compute_iou(pred_lanes, gt_lanes, y_samples, image_center_x=1280/2)