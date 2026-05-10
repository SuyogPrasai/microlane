import click, json
from pathlib import Path
from typing import List
from microlane.schemas.prediction import Evaluation
from scripts.core.evaluate import evaluate_scenario
from scripts.core.visualization import draw_evaluation_graphs
from scripts.core.results import load_results
from scripts.utils.evaluation_to_dic import evaluation_to_dict

@click.command()
@click.option('--path', '-p', required=True, help='Path to the scenario folder to evaluate')
@click.option('--save', '-s', is_flag=True, help='Whether to save the evaluation results')
@click.pass_context
def evaluate(ctx: click.Context, path: str, save: bool) -> None:

    scenario_path = Path(path)

    if scenario_path.is_dir() or not scenario_path.name.endswith('.json'):
        print(f"Error: {scenario_path} is not a valid prediction file")
        
    
    results: List[Evaluation] = evaluate_scenario(scenario_path)

    if save:

        output_file = scenario_path.parent / "evaluation.json"

        existing = []
        if output_file.exists():
            with open(output_file, 'r') as f:
                existing = json.load(f)

        existing.extend([evaluation_to_dict(r) for r in results])

        with open(output_file, 'w') as f:
            json.dump(existing, f, indent=4)

        print(f"Evaluation results saved to: {output_file}")

        load_results(output_file)

        draw_graphs = click.confirm("Do you want to draw the evaluation graphs?", default=True)

        if draw_graphs:
            draw_evaluation_graphs(output_file)