from typing import Tuple, List
from microlane.schema.sample import Sample
from microlane.schema.model_limbs import LaneNet2Input


class LaneNet2Formatter:
    
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size
    
    def format_one(self, sample: Sample) -> LaneNet2Input:
        pass
    
    def batch_format(self, samples: List[Sample]) -> List[LaneNet2Input]:
        return [self.format_one(sample) for sample in samples]