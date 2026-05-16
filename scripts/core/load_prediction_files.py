
import json
from pathlib import Path

from typing import Iterable, List

from microlane.schemas.evaluation import PredictionFile

def load_prediction_files(experiment_directory: Path | str) -> Iterable[PredictionFile]:
    
    experiment_directory = Path(experiment_directory)
    
    for directory in experiment_directory.rglob("*"):
        if not directory.is_dir():
            continue

        prediction_path = directory / "prediction.json"
        settings_path = directory / "settings.json"

        if not (prediction_path.exists() and settings_path.exists()):
            continue

        prediction = json.loads(prediction_path.read_text())
        settings = json.loads(settings_path.read_text())

        yield PredictionFile(
            file=prediction,
            model=settings["model"],
            dataset=settings["dataset"],
            augmentation=settings["augmentation"],
        )