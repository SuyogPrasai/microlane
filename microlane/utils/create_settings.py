import json
from pathlib import Path
from datetime import datetime

def create_settings(
    inference_vis_number: int,
    sample_number: int,
    model: str,
    dataset: str,
    augmentation_type: str,
    augmentation_settings: dict,
    output_path: str,
    experiment_name: str
) -> dict:
    
    settings = {
        "experiment": {
            "name": experiment_name,
            "sample_number": sample_number,
            "inference_vis_number": inference_vis_number,\
            "created_at": datetime.now().isoformat(),
        },
        "model": model,
        "dataset": dataset,
        "augmentation": {
            "type": augmentation_type,
            "settings": augmentation_settings,
        },
    }

    out = Path(output_path) / "settings.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(settings, f, indent=2)

    return settings