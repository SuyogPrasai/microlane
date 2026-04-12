import tensorflow as tf
import numpy as np
from typing import Tuple, List
import yaml, time


from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input
from microlane.schema.prediction import LanePrediction

from microlane.models.lanenet2.lanenet2.preprocessing import PreProcessor
from models.lanenet_lane_detection.lanenet_model import lanenet
from models.lanenet_lane_detection.local_utils.config_utils import parse_config_utils

class LaneNet2():
    
    def __init__(
        self,
        weights_path,
        config_path: str        
        ):
        
        self.weights_path = weights_path
        
        self.CFG = parse_config_utils.lanenet_cfg
                        
        self.preprocessor = PreProcessor(target_size=tuple(self.CFG['AUG']['EVAL_CROP_SIZE']))
                
        self._load_model()
             

    
    def infer(self, picture: Sample) -> LanePrediction:
        
        # I probably dont need the postprocessing step here since I am creating a unified preprocessing pipeline
                
        preprocessd_image = self.preprocessor.process_one(picture)
        
        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=self.weights_path)

            t_start = time.time()
            loop_times = 500
            binary_seg_image = None
            instance_seg_image = None
            for i in range(loop_times):
                binary_seg_image, instance_seg_image = self.sess.run(
                    [self.binary_seg_ret, self.instance_seg_ret],
                    feed_dict={self.input_tensor: [preprocessd_image]}
                )
            
            if binary_seg_image is not None:
                return LanePrediction(
                    binary_segmentation=binary_seg_image[0]
                )
                
            return LanePrediction(
                binary_segmentation=np.zeros((256, 512))
            )
        
    
    def batch_infer(self, batch: List[Sample]) -> List[LanePrediction]:
        """
        Prediction for a list of inputs
        
        """
        return [self.infer(item) for item in batch]
    
    def _load_config(self, config_path: str):
        
        # First Load the Configuation file
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
            
    
    def _load_model(self):
        
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        self.net = lanenet.LaneNet(phase='test', cfg=self.CFG)
        
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.sess_config = tf.ConfigProto()

        self.sess_config.gpu_options.per_process_gpu_memory_fraction = self.CFG.GPU.GPU_MEMORY_FRACTION

        self.sess_config.gpu_options.allow_growth = self.CFG.GPU.TF_ALLOW_GROWTH

        self.sess_config.gpu_options.allocator_type = 'BFC'
        
        self.sess = tf.Session(config=self.sess_config)


        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                self.CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        # define saver
        self.saver = tf.train.Saver(variables_to_restore)
