from typing import List

import numpy as np
import scipy.special  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]

from schemas.api_schemas import Sample, Prediction

from engine import UFLDEngine

class PostProcessor:

    def __init__(
        self,
        engine: UFLDEngine,
        target_size=(720, 1280),
    ) -> None:

        self.engine = engine

        self.img_h = target_size[0]
        self.img_w = target_size[1]

    def process(
        self,
        sample: Sample,
        output_tensor: torch.Tensor,
        run_time: float,
    ) -> Prediction:

        loc = self._postprocess(output_tensor)

        lanes = self._grid_to_lanes(
            loc,
            np.array(sample.h_samples),
        )

        return Prediction(
            samples=[sample],
            lanes=np.array(lanes),
            h_samples=np.array(sample.h_samples),
            run_time=float(run_time),
        )

    def _postprocess(
        self,
        raw_out: torch.Tensor,
    ) -> np.ndarray:

        out_j = raw_out[0].data.cpu().numpy()

        out_j = out_j[:, ::-1, :]

        prob = scipy.special.softmax(
            out_j[:-1, :, :],
            axis=0,
        )

        idx = (
            np.arange(self.engine.griding_num) + 1
        ).reshape(-1, 1, 1)

        loc = np.sum(
            prob * idx,
            axis=0,
        )

        argmax_j = np.argmax(
            out_j,
            axis=0,
        )

        loc[
            argmax_j == self.engine.griding_num
        ] = 0

        return loc

    def _grid_to_lanes(
        self,
        loc: np.ndarray,
        h_samples: np.ndarray,
    ) -> List[List[float]]:

        num_lanes = loc.shape[1]

        lanes: List[List[float]] = []

        for lane_idx in range(num_lanes):

            lane_col = loc[:, lane_idx]

            if np.sum(lane_col != 0) <= 2:
                continue

            pts = []

            for row_idx in range(
                self.engine.cls_num_per_lane
            ):

                if lane_col[row_idx] > 0:

                    x = int(
                        lane_col[row_idx]
                        * self.engine.col_sample_w
                        * self.img_w
                        / self.engine._NET_W
                    ) - 1

                    y = int(
                        self.img_h
                        * (
                            self.engine.row_anchor[
                                self.engine.cls_num_per_lane
                                - 1
                                - row_idx
                            ]
                            / self.engine._NET_H
                        )
                    ) - 1

                    pts.append((x, y))

            if not pts:

                lanes.append(
                    [-2] * len(h_samples)
                )

                continue

            pts_arr = np.array(
                pts,
                dtype=np.float32,
            )

            lane_xs: List[float] = []

            for y_target in h_samples.tolist():

                diff = np.abs(
                    pts_arr[:, 1] - y_target
                )

                if diff.min() > 10:

                    lane_xs.append(-2)

                else:

                    lane_xs.append(
                        int(
                            round(
                                pts_arr[
                                    np.argmin(diff),
                                    0,
                                ]
                            )
                        )
                    )

            lanes.append(lane_xs)

        return lanes