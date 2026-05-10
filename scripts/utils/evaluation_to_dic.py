from microlane.schemas.prediction import Evaluation

def evaluation_to_dict(e: Evaluation) -> dict:
    return {
        "run_time": e.predictions.run_time,
        "lanes": e.predictions.lanes,
        "h_samples": e.predictions.h_samples,
        "accuracy": e.accuracy,
        "precision": e.precision,
        "recall": e.recall,
        "f1_score": e.f1_score,
        "samples": [
            {
                "image_path": s.image_path,
                "dataset": s.dataset,
                "blur": s.blur,
                "lighting": s.lighting,
                "rotation": s.rotation,
                "zoom": s.zoom,
                "motion_blur": s.motion_blur,
            }
            for s in e.predictions.samples
        ]
    }
