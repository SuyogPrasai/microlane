import tensorflow as tf
import numpy as np
from typing import Tuple, List
import yaml, time


from microlane.models.lanenet2.lanenet2.schemas.api_schemas import Sample
from microlane.models.lanenet2.lanenet2.schemas.api_schemas import ModelPrediction

from microlane.models.lanenet2.lanenet2.helpers.preprocessing import PreProcessor
from microlane.models.lanenet2.lanenet2.engine import LaneNet2Engine

from lanenet_model import lanenet_postprocess # type: ignore
from local_utils.config_utils import parse_config_utils # type: ignore

CFG = parse_config_utils.lanenet_cfg

class LaneNet2():
    
    def __init__(
        self,
        weights_path,
        ):
        
        self.weights_path = weights_path
                                
        self.preprocessor = PreProcessor(target_size=(512, 256))
                             
        self._engine = LaneNet2Engine(weights_path)

    
    def infer(self, picture: Sample) -> ModelPrediction:
        
        # I probably dont need the postprocessing step here since I am creating a unified preprocessing pipeline
                
        processed_image = self.preprocessor.process(picture)
        
        if processed_image.image is None:
            raise ValueError(
                f"The processed image for sample '{picture.image_path}' is None. "
                "This should not happen. Please check the preprocessing step."
            )
            
        binary_seg, instance_seg, t_cost = self._engine.predict(processed_image.image)

        postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        postprocess_result = postprocessor.postprocess(
            
            binary_seg_result=binary_seg[0],
            
            instance_seg_result=instance_seg[0],
            
            source_image=picture.image.copy(),
            
            with_lane_fit=True, # Need to change don't know what does but true is default on the repo
            
            data_source='tusimple'
            
        )
        
                # mask_image = postprocess_result['mask_image']

        fit_params = postprocess_result['fit_params']

        h_samples = picture.h_samples
        
        lanes = []
        
        if fit_params is not None:
            
            remap_x = postprocessor._remap_to_ipm_x
            
            remap_y = postprocessor._remap_to_ipm_y
            
            ipm_h, ipm_w = remap_x.shape
            
            
            for fit_param in fit_params:
                
                fit_param = np.array(fit_param, dtype="float64")

                plot_y = np.linspace(10, ipm_h, ipm_h - 10)

                fit_x = fit_param[0] * (plot_y ** 2) + fit_param[1] * plot_y + fit_param[2] # type: ignore
                
                src_lane_pts = []
                
                for i in range(0, len(plot_y), 5):
                    ipm_xi = int(np.clip(fit_x[i], 0, ipm_w - 1))
                    ipm_yi = int(plot_y[i])
                    src_x = remap_x[ipm_yi, ipm_xi]
                    src_y = remap_y[ipm_yi, ipm_xi]
                    if src_x > 0 and src_y > 0:
                        src_lane_pts.append([src_x, src_y])

                if len(src_lane_pts) == 0:
                    lanes.append([-2] * len(h_samples))
                    continue

                pts = np.array(src_lane_pts, dtype="float32")  # Fix 2: string form instead of np.float32

                lane_xs = []

                for y in h_samples.tolist():
                    diff = np.abs(pts[:, 1] - y)
                    if diff.min() > 10:
                        lane_xs.append(-2)
                    else:
                        lane_xs.append(int(round(pts[np.argmin(diff), 0])))

                lanes.append(lane_xs)
                
        return ModelPrediction(
            sample=picture,
            lanes=lanes,
            run_time=t_cost
        )
    
    def batch_infer(self, batch: List[Sample]) -> List[ModelPrediction]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
    
    def close(self):
        self._engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()