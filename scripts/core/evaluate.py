import json
from typing import List
from pathlib import Path

from microlane.evalutation.lane_eval import LaneEval

def evaluate_scenario(scenario_path: Path, annotation_path: Path) -> List:
    
    output_folder = scenario_path.parent
    
    json_gt = [json.loads(line) for line in open(annotation_path)]
    
    json_pred = load_predictions(scenario_path)
    
    gt_by_file = {gt['raw_file']: gt for gt in json_gt}
    
    results = []

    for prediction in json_pred:
        
        raw_file = prediction['raw_file']
        
        gt = gt_by_file.get(raw_file)
        
        if gt is None:
            
            print(f"No ground truth found for: {raw_file}")
            
            continue
        
        pred_lanes = prediction['lanes']
        
        run_time = prediction['run_time']
        
        gt_lanes = gt['lanes']
        
        h_samples = gt['h_samples']
        
        accuracy, fp, fn = LaneEval.bench(pred_lanes, gt_lanes, h_samples, run_time)

        precision = 1 - fp
        
        recall = 1 - fn
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'raw_file': raw_file,
            'accuracy': accuracy,
            'fp': fp,
            'fn': fn,
            'run_time': run_time,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
    return results

def load_predictions(predictions_path: Path) -> List:
    
    with open(predictions_path, "r") as file:
        
        content = file.read().strip()
    
        json_pred: List = json.loads(content)
        
    return json_pred