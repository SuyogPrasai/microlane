import numpy as np
from typing import List

from microlane.models.ufld.ufld.schemas.api_schemas import Sample
from microlane.models.ufld.ufld.schemas.api_schemas import ModelPrediction

from microlane.models.ufld.ufld.helpers.preprocessing import PreProcessor
from microlane.models.ufld.ufld.engine import UFLDEngine

class UFLD():
    
    def __init__(self, weights_path, dataset: str = "tusimple") -> None:
        
        self._engine = UFLDEngine(weights_path)
        
        self.preprocessor = PreProcessor(target_size=(512, 256))
        
        # Convenience aliases surfaced from the engine for post-processing
        self._img_w = 1280 if dataset.lower() == "tusimple" else 1640

        self._img_h = 720  if dataset.lower() == "tusimple" else 590

    
    def infer(self, picture: Sample ) -> ModelPrediction:
        
        processed_image = self.preprocessor.process(picture)
        
        if processed_image.image is None:
            raise ValueError(
                f"Processed image for sample '{picture.image_path}' is None. "
                "Please check the preprocessing step."
            )

        out_j, t_cost = self._engine.predict(processed_image.image)

        lanes = self._grid_to_lanes(out_j, picture.h_samples)

        return ModelPrediction(
            sample=picture,
            lanes=lanes,
            run_time=t_cost,
        )
    
    
    def _grid_to_lanes(
        self,
        out_j: np.ndarray,
        h_samples: np.ndarray,
    ) -> List[List[float]]:
        
        engine = self._engine

        col_sample_w = engine.col_sample_w

        row_anchor = engine.row_anchor
        
        cls_num = engine.cls_num_per_lane
        
        img_w = self._img_w

        img_h = self._img_h
        
        num_lanes = out_j.shape[1]
        
        lanes: List[List[float]] = []

        for lane_idx in range(num_lanes):
            
            lane_col = out_j[:, lane_idx]

            # Skip lanes where fewer than 2 rows are active
            if np.sum(lane_col != 0) <= 2:
                continue

            # Build (x, y) pixel pairs for every active row anchor
            pts = []
            for row_idx in range(cls_num):
                if lane_col[row_idx] > 0:
                    x = int(
                        lane_col[row_idx] * col_sample_w * img_w / 800
                    ) - 1
                    y = int(
                        img_h * (row_anchor[cls_num - 1 - row_idx] / 288)
                    ) - 1
                    pts.append((x, y))

            if not pts:
                lanes.append([-2] * len(h_samples))
                continue

            pts_arr = np.array(pts, dtype="float32")  # (N, 2)  — columns: x, y

            # For every requested h_sample row, find the closest detected y
            lane_xs: List[float] = []
            for y_target in h_samples.tolist():
                diff = np.abs(pts_arr[:, 1] - y_target)
                if diff.min() > 10:          # too far → no detection
                    lane_xs.append(-2)
                else:
                    lane_xs.append(int(round(pts_arr[np.argmin(diff), 0])))

            lanes.append(lane_xs)

        return lanes
    
    def close(self):
        self._engine.close()
    
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()