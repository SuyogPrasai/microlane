import click
import json
from pathlib import Path

from scripts.core.evaluate import evaluate_scenario
from scripts.core.visualization import draw_evaluation_graphs
from scripts.core.results import load_results


@click.command()
@click.option(
    "--progress",
    "-p",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to progress.json",
)
@click.option(
    "--save",
    "-s",
    is_flag=True,
    help="Save evaluation results to evaluation.json in each run directory",
)
@click.option(
    "--graphs",
    "-g",
    is_flag=True,
    help="Draw graphs for runs missing them, even if evaluation.json already exists",
)
@click.option(
    "--graph-re",
    "-gr",
    is_flag=True,
    help="Regenerate graphs for all runs, even if they already exist",
)
@click.option(
    "--light-theme",
    "-lt",
    is_flag=True,
    help="Generate graphs in light theme",
)
@click.pass_context
def assess(
    ctx: click.Context,
    progress: Path,
    save: bool,
    graphs: bool,
    graph_re: bool,
    light_theme: bool,
) -> None:
    """
    Read progress.json and:
      - Evaluate runs missing evaluation.json (requires --save to persist)
      - Draw graphs for runs missing them when --graphs is passed
      - Regenerate graphs for all runs when --graph-re is passed
      - Use light theme for graphs when --light-theme is passed
    """
    datasets_config = ctx.obj["config"]["data"]["datasets"]

    with open(progress) as f:
        progress_data = json.load(f)

    missing_eval: list[tuple[str, Path, str]] = []  # needs evaluation
    missing_graphs: list[tuple[str, Path]] = []      # has evaluation, needs graphs

    index = progress_data.get("index", {})
    for dataset, models in index.items():
        for model, conditions in models.items():
            for condition, runs in conditions.items():
                for run_id, run_data in runs.items():
                    label = f"{dataset}/{model}/{condition}/{run_id}"
                    run_dir = Path(run_data["run_dir"])
                    issues = run_data.get("health", {}).get("issues", [])
                    counts = run_data.get("counts", {})

                    if "missing evaluation.json" in issues:
                        missing_eval.append((label, run_dir / "prediction.json", dataset))
                    elif graphs or graph_re:
                        no_distribution = counts.get("distribution_plots", 0) == 0
                        no_progression = counts.get("progression_plots", 0) == 0
                        if no_distribution or no_progression or graph_re:
                            missing_graphs.append((label, run_dir / "evaluation.json"))

    # ── Evaluate missing runs ────────────────────────────────────────────────
    if missing_eval:
        click.echo(f"Found {len(missing_eval)} run(s) missing evaluation.json:\n")
        for label, _, _ in missing_eval:
            click.echo(f"  • {label}")
        click.echo()

        for label, prediction_path, dataset in missing_eval:
            click.echo(f"Evaluating: {label}")

            if dataset not in datasets_config:
                click.echo(f"  ✗ Dataset '{dataset}' not found in config, skipping.\n")
                continue

            annotation_path = Path(datasets_config[dataset]["annotation_file"])

            if not prediction_path.exists():
                click.echo(f"  ✗ prediction.json not found at {prediction_path}, skipping.\n")
                continue

            try:
                results = evaluate_scenario(prediction_path, annotation_path)
            except Exception as e:
                click.echo(f"  ✗ evaluate_scenario failed: {e}\n")
                continue

            if not save:
                click.echo(f"  ✓ Evaluated {len(results)} samples (dry run, not saved).\n")
                continue

            output_file = prediction_path.parent / "evaluation.json"

            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            click.echo(f"  ✓ Saved → {output_file}")

            try:
                load_results(output_file)
            except Exception as e:
                click.echo(f"  ⚠ load_results failed: {e}")

            if graphs or graph_re:
                try:
                    draw_evaluation_graphs(output_file, light_theme=light_theme)
                    click.echo(f"  ✓ Graphs drawn.")
                except Exception as e:
                    click.echo(f"  ⚠ draw_evaluation_graphs failed: {e}")

            click.echo()
    else:
        click.echo("No runs missing evaluation.json.")

    # ── Draw graphs for runs that already have evaluation but no/outdated graphs ──
    if graphs or graph_re:
        if missing_graphs:
            if graph_re:
                click.echo(f"\nRegenerating graphs for {len(missing_graphs)} run(s):\n")
            else:
                click.echo(f"\nFound {len(missing_graphs)} run(s) missing graphs:\n")

            for label, _ in missing_graphs:
                click.echo(f"  • {label}")
            click.echo()

            for label, evaluation_path in missing_graphs:
                click.echo(f"Drawing graphs: {label}")

                if not evaluation_path.exists():
                    click.echo(f"  ✗ evaluation.json not found at {evaluation_path}, skipping.\n")
                    continue

                try:
                    draw_evaluation_graphs(evaluation_path, light_theme=light_theme)
                    click.echo(f"  ✓ Graphs drawn.\n")
                except Exception as e:
                    click.echo(f"  ⚠ draw_evaluation_graphs failed: {e}\n")
        else:
            click.echo("No runs missing graphs.")

    click.echo("\nDone.")