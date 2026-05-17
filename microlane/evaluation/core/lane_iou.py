import numpy as np
from itertools import combinations
from shapely.geometry import Polygon

class LaneIoU:

    @staticmethod
    def lane_to_points(x_coords, y_samples):
        """
        Filter out invalid (negative) x points and return (x, y) pairs.
        """
        points = [(x, y) for x, y in zip(x_coords, y_samples) if x >= 0]
        return points

    @staticmethod
    def build_polygon(lane_a, lane_b, y_samples):
        """
        Build a closed polygon from two lanes.
        Traverse lane_a top→bottom, then lane_b bottom→top.
        Returns a Shapely Polygon or None if not enough points.
        """
        pts_a = LaneIoU.lane_to_points(lane_a, y_samples)
        pts_b = LaneIoU.lane_to_points(lane_b, y_samples)

        if len(pts_a) < 2 or len(pts_b) < 2:
            return None

        # Polygon: left side top→bottom, right side bottom→top
        poly_points = pts_a + pts_b[::-1]

        try:
            poly = Polygon(poly_points)
            if not poly.is_valid:
                poly = poly.buffer(0)  # fix self-intersections
            return poly
        except Exception:
            return None

    @staticmethod
    def best_pair_polygon(lanes, y_samples):
        """
        From a list of lanes, find the pair that encloses the largest area.
        Returns the best Polygon or None.
        """
        best_poly = None
        best_area = 0.0

        for lane_a, lane_b in combinations(lanes, 2):
            poly = LaneIoU.build_polygon(lane_a, lane_b, y_samples)
            if poly is None:
                continue
            area = poly.area
            if area > best_area:
                best_area = area
                best_poly = poly

        return best_poly

    @staticmethod
    def compute_iou(pred, gt, y_samples):
        """
        Compute IoU between the best-pair polygon of pred lanes
        and the best-pair polygon of gt lanes.
        Returns float in [0, 1]. Returns 0 if either polygon is None.
        """
        gt_poly   = LaneIoU.best_pair_polygon(gt,   y_samples)
        pred_poly = LaneIoU.best_pair_polygon(pred,  y_samples)

        if gt_poly is None or pred_poly is None:
            return 0.0

        intersection = gt_poly.intersection(pred_poly).area
        union        = gt_poly.union(pred_poly).area

        if union == 0:
            return 0.0

        return intersection / union