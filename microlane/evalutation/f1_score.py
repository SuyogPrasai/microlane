from typing import List, Tuple

from microlane.schemas.prediction import Prediction, Evaluation

def calculate_f1_score(prediction: Prediction) -> Tuple[float, float, float]:
    
    f1_score = 0.0
    
    precision = 0.0
    
    recall = 0.0

    return f1_score, precision, recall