
from typing import Tuple, List


class TuSimple():    
    """
    Class structuring for the tusimple datset

    """
    
    def __init__(
        self, 
        dimensions: Tuple[int, int],
        path: str,
        annotation_file_path: str
        ) -> None:
        
        self.image_dimensions = dimensions
        self.folder_location = path
        self.annotation_file_path = annotation_file_path