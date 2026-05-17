## Append a Evaluation onto a csv file

import csv
from pathlib import Path

from microlane.schemas.evaluation import Evaluation

FIELDS = [
    "experiment_number", "dataset", "model", "augmentation",
    "raw_file", "processed_samples", "run_time",
    "accuracy", "IOU", "fn", "fp"
]

def store_evaluation(
    evaluation: Evaluation,
    csv_path: Path | str
):
    """
    Append a single Evaluation row to a CSV file.
    Creates a file with header if it doesn't exist
    """
    
    csv_path = Path(csv_path)
    
    file_exists = csv_path.exists()
    
    with csv_path.open("a", newline="") as f:
        
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            
            **{k: getattr(evaluation, k) for k in FIELDS if k != "processed_samples"},
            "processed_samples": ";".join(evaluation.processed_samples),
        })