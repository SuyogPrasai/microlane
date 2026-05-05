# Produce Evaluation Objects from a Given set of Predictions
from typing import List

from microlane.schemas.prediction import Prediction, Evaluation
from microlane.evalutation.accuracy import calculate_accuracy
from microlane.evalutation.f1_score import calculate_f1_score

class Evaluator:
    
    def __init__(self, predictions: List[Prediction]):
        
        self.predictions = predictions
        
    
    def evaluate(self) -> List[Evaluation]:
        
        evaluations: List[Evaluation] = []
        
        for prediction in self.predictions:
            
            accuracy = calculate_accuracy(prediction)
            
            f1_score, precision, recall = calculate_f1_score(prediction)
            
            evaluation = Evaluation(
                predictions=prediction,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            evaluations.append(evaluation)
        
        return evaluations