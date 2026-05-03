import click, json
from pathlib import Path
from scripts.core.evaluate import evaluate_scenario
from scripts.core.visualization import draw_evaluation_graphs
from scripts.core.results import load_results

@click.command()
@click.option('--path', '-p', required=True, help='Path to the scenario folder to evaluate')
@click.option('--save', '-s', is_flag=True, help='Whether to save the evaluation results')
@click.pass_context
def evaluate(ctx: click.Context, path: str, save: bool) -> None:
    
    scenario_path = Path(path)
    
    if scenario_path.is_dir() or not scenario_path.name.endswith('.json'):
        print(f"Error: {scenario_path} is not a valid prediction file")
        return
    
    with open(scenario_path) as f:
        predictions = json.load(f)
    
    sample_path = predictions[0]["raw_file"]
    dataset = "raw" if "13-minute-sampling" in sample_path else "tusimple"
    annotation_path = Path(ctx.obj["config"]["data"]["datasets"][dataset]["annotation_file"])
    
    results = evaluate_scenario(scenario_path, annotation_path)
    
    if save:
        
        output_file = scenario_path.parent / "evaluation.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Evaluation results saved to: {output_file}")
        
        load_results(output_file)
        
        draw_graphs = click.confirm("Do you want to draw the evaluation graphs?", default=True)
        
        if draw_graphs:
            draw_evaluation_graphs(output_file)