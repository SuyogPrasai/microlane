from schemas.api_schemas import Sample  # pyright: ignore[reportMissingImports]
from schemas.api_schemas import Prediction  # pyright: ignore[reportMissingImports]

from helpers.preprocessing import PreProcessor # pyright: ignore[reportMissingImports]
from helpers.postprocessing import PostProcessor # pyright: ignore[reportMissingImports]
from engine import LaneNetEngine  # pyright: ignore[reportMissingImports]


class LaneNet():
    
    def __init__(self, weights_path) -> None:
        
        self.weights_path = weights_path
        
        self.preprocessor = PreProcessor()
        
        self.postprocessor = PostProcessor()
        
        self.engine = LaneNetEngine(
            
            weights_path=weights_path
        )        
    
    def infer(self, picture: Sample) -> Prediction:
        
        processed_image = self.preprocessor.process(picture)
        
        if processed_image.image is None:
            raise ValueError(
                f"The processed image for sample '{picture.image_path}' is None. "
                "This should not happen. Please check the preprocessing step."
            )
            
        binary_seg, instance_seg, t_cost = self.engine.predict(processed_image.image)
        
        return self.postprocessor.process(
            sample=picture,
            binary_segmentation=binary_seg,
            instance_segmentation=instance_seg,
            run_time=t_cost
        )