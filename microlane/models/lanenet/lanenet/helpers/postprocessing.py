from typing import Tuple, List
from schemas.api_schemas import Sample, Prediction
import numpy as np
import cv2

from lanenet_model import lanenet_postprocess # type: ignore
from local_utils.config_utils import parse_config_utils # type: ignore

CFG = parse_config_utils.lanenet_cfg

class PostProcessor():
    
    def __init__(self) -> None:
        
        self.target_size = [720, 1280]
        
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        
    def process(self, 
                sample: Sample, 
                binary_segmentation: np.ndarray,
                instance_segmentation: np.ndarray,
                run_time: float
                ) -> Prediction:
        
        H, W = self.target_size
        
        postprocess_result = self.postprocessor.postprocess(
            
            binary_seg_result=binary_segmentation[0],
            
            instance_seg_result=instance_segmentation[0],
            
            source_image=sample.image.copy(),
            
            with_lane_fit=True, # Need to change don't know what does but true is default on the repo
            
            data_source='tusimple'
            
        )
        
        fit_params = postprocess_result['fit_params']

        h_samples = sample.h_samples
        
        lanes = []

        
        if fit_params is not None:
            
            remap_x = self.postprocessor._remap_to_ipm_x
            
            remap_y = self.postprocessor._remap_to_ipm_y
            
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
        
        return Prediction(
            samples=[sample],
            lanes=np.array(lanes),
            h_samples=np.array(h_samples),
            run_time=float(run_time)
        )