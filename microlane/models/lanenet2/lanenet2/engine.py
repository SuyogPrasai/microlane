import numpy as np
import tensorflow as tf
from typing import Tuple

from lanenet_model import lanenet # type: ignore
from local_utils.config_utils import parse_config_utils # type: ignore
from local_utils.log_util import init_logger # type: ignore

class LaneNet2Engine():
    
    def __init__(self, weights_path: str) -> None:
        
        self.weights_path = weights_path
        
        self.CFG = parse_config_utils.lanenet_cfg
        
       
        # Initializing the initial parameters for the engine
         
        self._input_tensor = None

        self._binary_seg_ret = None

        self._instance_seg_ret = None

        self._sess = None
        
        self._build_graph()
    
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single forward pass.

        Args:
            preprocessed_image: float32 array of shape (256, 512, 3),
                                 already normalised to [-1, 1].

        Returns:
            binary_seg:   shape (1, 256, 512)
            instance_seg: shape (1, 256, 512, EMBEDDING_FEATS_DIMS)
        """
        
        if self._sess is None:
            
            raise RuntimeError("Session is not initialised. Call _build_graph() first.")

        binary_seg, instance_seg = self._sess.run(
            
            [self._binary_seg_ret, self._instance_seg_ret],
            
            feed_dict={self._input_tensor: [image]},

        )
        
        return binary_seg, instance_seg
    
    def close(self) -> None:
        """
        Clean up resources, close the TensorFlow session.
        
        """
        
        if self._sess is not None:
            
            self._sess.close()
            
            self._sess = None

            
    def _build_graph(self) -> None:
        
        # Define the input tensor
        
        self._input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[1, 256, 512, 3],
            name='input_tensor',
        )

        net = lanenet.LaneNet(phase='test', cfg=self.CFG)
    
        self._binary_seg_ret, self._instance_seg_ret = net.inference(
            input_tensor=self._input_tensor,
            name='LaneNet',
        )
        
        # Initializing Sess and Loading Weights
        
        sess_config = tf.ConfigProto()

        sess_config.gpu_options.per_process_gpu_memory_fraction = self.CFG.GPU.GPU_MEMORY_FRACTION

        sess_config.gpu_options.allow_growth = self.CFG.GPU.TF_ALLOW_GROWTH

        sess_config.gpu_options.allocator_type = 'BFC'

        self._sess = tf.Session(config=sess_config)
        
        
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(self.CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            
        saver = tf.train.Saver(variables_to_restore)
        
        saver.restore(sess=self._sess, save_path=self.weights_path)