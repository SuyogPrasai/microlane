"""
Code which helps us work with the TuSimple Dataset localted at data/TuSimple
"""



from typing import Tuple, List

# This is a common dataset example that can be implemented for all types, right now i have only done for TUsimple
# Buyt it may be more appropriate to create a Dataset Class and extend it for TuSimple, and add more properties accordingly

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