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
    
    annotation_path = Path(ctx.obj["config"]["data"]["datasets"]["tusimple"]["annotation_file"])
    
    results = evaluate_scenario(scenario_path, annotation_path) # Generate the evaluation and store
    
    if save:
        
        output_file = scenario_path.parent / "evaluation.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Evaluation results saved to: {output_file}")
    
        # Store overall average results along with the top 5 and bottom 5 performing scenarios
        
        load_results(output_file) # Load the results and store the summary
        

        draw_graphs = click.confirm("Do you want to draw the evaluation graphs?", default=True)
        
        if draw_graphs:

            draw_evaluation_graphs(output_file)