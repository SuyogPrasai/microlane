import tensorflow as tf
import numpy as np
from typing import Tuple, List

from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
from microlane.schema.prediction import LanePrediction
from utils import parse_config_utils


CFG = parse_config_utils.lanenet_cfg

class LaneNet2():
    
    def __init__(
        self,
        weights_path,
        with_lane_fit: bool = True,
        
        ):
        
        self.weights_path = weights_path
        self.with_lane_fit = with_lane_fit

        self._input_tensor = None
        self._binary_seg_ret = None
        self._instance_seg_ret = None
        self._sess = None
        self._postprocessor = None
        
        self._load_model()
             

    
    def infer(self, image: LaneNet2Input) -> LanePrediction:
           
        pass
        
    
    def batch_infer(self, batch: List[LaneNet2Input]) -> List[LanePrediction]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
    
    
    def _load_model(self):
        
        self._input_tensor = tf.placeholder(
            dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor'
        )
        
        
        
    
    @staticmethod
    def _minmax_scale(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) * 255.0 / (mx - mn)
  
    