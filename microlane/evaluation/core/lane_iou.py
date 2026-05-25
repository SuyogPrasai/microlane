import numpy as np
from itertools import combinations
from shapely.geometry import Polygon

class LaneIoU:

    @staticmethod
    def lane_to_points(x_coords, y_samples):
        return [(x, y) for x, y in zip(x_coords, y_samples) if x >= 0]

    @staticmethod
    def _lane_mean_x(lane, y_samples):
        xs = [x for x, _ in LaneIoU.lane_to_points(lane, y_samples)]
        return np.mean(xs) if xs else None
    
    @staticmethod
    def build_polygon(lane_a, lane_b, y_samples):
        pts_a = LaneIoU.lane_to_points(lane_a, y_samples)
        pts_b = LaneIoU.lane_to_points(lane_b, y_samples)

        if len(pts_a) < 2 or len(pts_b) < 2:
            return None

        poly_points = pts_a + pts_b[::-1]

        try:
            poly = Polygon(poly_points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly
        except Exception:
            return None
    
    @staticmethod
    def ego_pair_polygon(lanes, y_samples, image_center_x):
        left_candidate = None
        right_candidate = None
        
        for lane in lanes:
            mx = LaneIoU._lane_mean_x(lane, y_samples)
            if mx is None:
                continue
            
            if mx < image_center_x:
                if left_candidate is None or mx > left_candidate[0]:
                    left_candidate = (mx, lane)
            else:
                if right_candidate is None or mx < right_candidate[0]:
                    right_candidate = (mx, lane)        
        
        if left_candidate is None or right_candidate is None:
            return None

        return LaneIoU.build_polygon(
            
            left_candidate[1], right_candidate[1], y_samples
        )
    
    
    @staticmethod
    def compute_iou(
        pred, gt, y_samples, image_center_x
    ):
    
        gt_poly   = LaneIoU.ego_pair_polygon(gt,   y_samples, image_center_x)
        pred_poly = LaneIoU.ego_pair_polygon(pred,  y_samples, image_center_x)

        polygon_ok = (
            gt_poly is not None and pred_poly is not None
            and gt_poly.area > 0  and pred_poly.area > 0
        )

        if polygon_ok:
            intersection = gt_poly.intersection(pred_poly).area # type: ignore
            union        = gt_poly.union(pred_poly).area # type: ignore
            return 0.0 if union == 0 else intersection / union

        return 0.0