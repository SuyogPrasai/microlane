import json
from typing import List
from pathlib import Path
import cv2
import numpy as np

from microlane.evalutation.evaluator import Evaluator
from microlane.schemas.prediction import Prediction, Evaluation
from microlane.schemas.sample import Sample


def evaluate_scenario(scenario: Path) -> List[Evaluation]:

    predictions = load_predictions(scenario)
    
    evaluator = Evaluator(predictions)
            
    evaluations = evaluator.evaluate()
            
    return evaluations

def load_predictions(predictions_path: Path) -> List[Prediction]:
    with open(predictions_path) as f:
        data = json.loads(f.read())

    if isinstance(data, dict):
        data = [data]

    return [
        Prediction(
            lanes=d['lanes'],
            h_samples=d['h_samples'],
            run_time=d['run_time'],
            samples=[s for s in (_load_sample(raw) for raw in d.get('samples', [])) if s is not None]
        )
        for d in data
    ]


def _load_sample(s: dict) -> Sample | None:
    image = cv2.imread(s['image_path'])
    if image is None:
        return None
    
    assert isinstance(image, np.ndarray)
    
    return Sample(
        image=image,
        image_path=s['image_path'],
        lanes=s['lanes'],
        h_samples=s['h_samples'],
        dataset=s['dataset'],
        blur=s.get('blur', 0.0),
        lighting=s.get('lighting', 1.0),
        rotation=s.get('rotation', 0.0),
        zoom=s.get('zoom', 1.0),
        motion_blur=s.get('motion_blur', 0.0)
    )