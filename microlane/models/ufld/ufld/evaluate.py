import torch  # pyright: ignore[reportMissingImports]

from schemas.api_schemas import Sample, Prediction

from helpers.preprocessing import PreProcessor
from helpers.postprocessing import PostProcessor
from engine import UFLDEngine


class UFLD:

    def __init__(self, weights_path: str):

        self.engine = UFLDEngine(weights_path)

        self.preprocessor = PreProcessor(self.engine)

        self.postprocessor = PostProcessor(self.engine)

    def infer(self, picture: Sample) -> Prediction:

        processed_image: torch.Tensor = self.preprocessor.process(picture)

        out_j, t_cost = self.engine.predict(processed_image)

        return self.postprocessor.process(
            sample=picture,
            output_tensor=out_j,
            run_time=t_cost,
        )