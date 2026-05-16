# Produce Evaluation Objects from a Given set of Predictions

from microlane.schemas.prediction import Prediction
from microlane.schemas.evaluation import Evaluation

from microlane.evaluation.tusimple_benchmark import calculate_f1_score
from microlane.evaluation.iou import calculate_iou
   
    
def evaluate_prediction( 
            prediction: Prediction, 
            experiment_number: int,
            model: str,
            dataset: str,
            augmentation: str
            ) -> Evaluation:
            
    accuracy, f1_score, precision, recall = calculate_f1_score(prediction)
    
    iou = calculate_iou(prediction)
    
    evaluation = Evaluation(
        experiment_number=experiment_number,
        dataset=dataset,
        model=model,
        augmentation=augmentation,
        raw_file=prediction.samples[-1].image_path,
        processed_samples = [sample.image_path for sample in prediction.samples],
        run_time=prediction.run_time,
        accuracy=accuracy,
        f1_score=f1_score,
        IOU=iou,
        precision=precision,
        recall=recall            
    )
                
    return evaluation