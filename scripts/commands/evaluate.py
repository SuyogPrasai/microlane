import click

from microlane.schemas.prediction import Prediction
from microlane.schemas.sample import Sample

from microlane.evaluation.evaluator import evaluate_prediction

from scripts.core.load_prediction_files import load_prediction_files
from scripts.core.read_image import read_image
from scripts.core.evaluation_to_csv import store_evaluation

@click.command()
@click.option('--path', '-p', required=True, help='Path to the experiment folder to evaluate')
@click.option('--csv', '-c', required=True, help='Path to the CSV file where results are stored')
def evaluate(path: str, csv: str):
    
    print("\n")
    experiment_number = 1
    
    for prediction_file in load_prediction_files(experiment_directory=path):
        
        prediction_data = prediction_file.file
        model = prediction_file.model
        dataset = prediction_file.dataset
        augmentation = prediction_file.augmentation["type"]

        evaluations_in_file = 0

        with click.progressbar(
            prediction_data,
            label=click.style(f"{model} | {dataset} | {augmentation}", fg="cyan"),
            show_pos=True,
        ) as predictions:
            for prediction in predictions:
                
                prediction_object = Prediction(
                    lanes=prediction["lanes"],
                    h_samples=prediction["h_samples"],
                    run_time=prediction["run_time"],
                    samples=[
                        Sample(
                            image_path=sample["image_path"],
                            image=read_image(sample["image_path"]),
                            lanes=sample["lanes"],
                            h_samples=sample["h_samples"],
                            dataset=sample["dataset"],
                            blur=sample["blur"],
                            lighting=sample["lighting"],
                            rotation=sample["rotation"],
                            zoom=sample["zoom"],
                            motion_blur=sample["motion_blur"],
                        )
                        for sample in prediction["samples"]
                    ]
                )
                
                evaluation = evaluate_prediction(
                    prediction=prediction_object,
                    experiment_number=experiment_number,
                    model=model,
                    dataset=dataset,
                    augmentation=augmentation
                )
                
                experiment_number += 1
                evaluations_in_file += 1
                store_evaluation(evaluation=evaluation, csv_path=csv)