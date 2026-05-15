from typing import List

import numpy as np, cv2
import torch # pyright: ignore[reportMissingImports]

from schemas.api_schemas import Prediction, Sample

class PostProcessor:

    def __init__(self) -> None:
        pass

    def process(
        self,
        samples: List[Sample],
        run_time: float,
        mask,
    ) -> Prediction:

        ref_image = samples[-1]

        orig_h, orig_w = ref_image.image.shape[:2]

        h_samples = ref_image.h_samples

        mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        lanes = self._mask_to_lanes(mask_full, h_samples)

        return Prediction(
            samples=samples,
            lanes=lanes,
            h_samples=samples[-1].h_samples,
            run_time=run_time
        )


    def _mask_to_lanes(self, mask: np.ndarray, h_samples: np.ndarray) -> np.ndarray:

        row_centroids = []

        for y in h_samples:
            row     = mask[int(y), :]
            changes = np.diff(row.astype(np.int8), prepend=0, append=0)
            starts  = np.where(changes == 1)[0]
            ends    = np.where(changes == -1)[0]
            row_centroids.append(sorted((starts + ends) / 2.0))

        max_lanes = max((len(r) for r in row_centroids), default=0)

        if max_lanes == 0:
            return np.full((0, len(h_samples)), -2, dtype=np.float32)

        n_rows = len(h_samples)
        order  = np.argsort(h_samples)[::-1]

        tracks  = {}
        next_id = 0
        MAX_DIST = 50

        for row_idx in order:
            centroids = row_centroids[row_idx]

            if not centroids:
                continue

            if not tracks:
                for x in centroids:
                    tracks[next_id] = [(row_idx, x)]
                    next_id += 1
                continue

            last_x = {lid: pts[-1][1] for lid, pts in tracks.items()}

            used_tracks = set()
            used_cents  = set()
            pairs       = []

            for c_i, cx in enumerate(centroids):
                best_lid, best_d = None, MAX_DIST

                for lid, lx in last_x.items():
                    if lid in used_tracks:
                        continue
                    d = abs(cx - lx)
                    if d < best_d:
                        best_d, best_lid = d, lid

                if best_lid is not None:
                    pairs.append((best_lid, c_i, cx))
                    used_tracks.add(best_lid)
                    used_cents.add(c_i)

            for lid, _, cx in pairs:
                tracks[lid].append((row_idx, cx))

            for c_i, cx in enumerate(centroids):
                if c_i not in used_cents:
                    tracks[next_id] = [(row_idx, cx)]
                    next_id += 1

        lane_array = np.full((len(tracks), n_rows), -2, dtype=np.float32)

        for lane_out_idx, (lid, pts) in enumerate(
            sorted(tracks.items(), key=lambda kv: np.mean([x for _, x in kv[1]]))
        ):
            for row_idx, x in pts:
                lane_array[lane_out_idx, row_idx] = x

        return lane_array