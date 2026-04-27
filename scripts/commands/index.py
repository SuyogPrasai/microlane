import click
import json
from pathlib import Path
from datetime import datetime


def count_prediction_samples(prediction_path: Path) -> int | None:
    """Count number of samples in prediction.json."""
    try:
        with open(prediction_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            for key in ("predictions", "samples", "results"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
            return len(data)
    except Exception:
        return None


def parse_evaluation(evaluation_path: Path) -> dict:
    """
    Parse evaluation.json — a flat list of per-sample result dicts.
    Returns sample count and mean of each metric.
    """
    try:
        with open(evaluation_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {"ok": False, "error": f"expected list, got {type(data).__name__}"}

        count = len(data)
        metrics = ["accuracy", "fp", "fn", "precision", "recall", "f1", "run_time"]
        summary: dict[str, float] = {}
        for metric in metrics:
            values = [s[metric] for s in data if metric in s]
            if values:
                summary[metric] = round(sum(values) / len(values), 6)

        return {
            "ok": True,
            "sample_count": count,
            "summary": summary,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def audit_run(run_dir: Path) -> dict:
    """Audit a single run directory."""
    prediction_path = run_dir / "prediction.json"
    evaluation_path = run_dir / "evaluation.json"
    settings_path = run_dir / "settings.json"
    inference_dir = run_dir / "inference"
    distribution_dir = run_dir / "distribution"
    progression_dir = run_dir / "progression"

    prediction_count = (
        count_prediction_samples(prediction_path)
        if prediction_path.exists()
        else None
    )

    has_evaluation = evaluation_path.exists()
    evaluation = parse_evaluation(evaluation_path) if has_evaluation else None

    health: dict = {"ok": True, "issues": []}

    if not prediction_path.exists():
        health["issues"].append("missing prediction.json")
        health["ok"] = False

    if not has_evaluation:
        health["issues"].append("missing evaluation.json")
        health["ok"] = False
    elif evaluation is None:
        health["issues"].append("evaluation.json could not be read")
        health["ok"] = False
    elif not evaluation["ok"]:
        health["issues"].append(f"evaluation.json parse error: {evaluation['error']}")
        health["ok"] = False
    else:
        eval_count = evaluation["sample_count"]
        if prediction_count is not None and eval_count != prediction_count:
            health["issues"].append(
                f"sample count mismatch: prediction has {prediction_count}, "
                f"evaluation has {eval_count}"
            )
            health["ok"] = False

    inference_count = (
        len(list(inference_dir.glob("*.png"))) if inference_dir.exists() else 0
    )
    distribution_count = (
        len(list(distribution_dir.glob("*.png"))) if distribution_dir.exists() else 0
    )
    progression_count = (
        len(list(progression_dir.glob("*.png"))) if progression_dir.exists() else 0
    )

    result: dict = {
        "run_dir": str(run_dir),
        "health": health,
        "counts": {
            "prediction_samples": prediction_count,
            "evaluation_samples": evaluation["sample_count"] if (evaluation and evaluation["ok"]) else None,
            "inference_visualizations": inference_count,
            "distribution_plots": distribution_count,
            "progression_plots": progression_count,
        },
        "files": {
            "has_prediction": prediction_path.exists(),
            "has_evaluation": has_evaluation,
            "has_settings": settings_path.exists(),
            "has_cumulative_mean": (run_dir / "cumulative_mean.png").exists(),
        },
    }

    if has_evaluation and evaluation is not None and evaluation["ok"]:
        result["evaluation"] = {
            "sample_count": evaluation["sample_count"],
            "summary": evaluation["summary"],
        }

    return result


def build_index(results_dir: Path) -> dict:
    """
    Walk results_dir and build a nested index:
    dataset -> model -> condition -> run_timestamp -> audit
    """
    index: dict = {}

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "progress.json":
            continue
        dataset = dataset_dir.name
        index[dataset] = {}

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            index[dataset][model] = {}

            for condition_dir in sorted(model_dir.iterdir()):
                if not condition_dir.is_dir():
                    continue
                condition = condition_dir.name
                index[dataset][model][condition] = {}

                for run_dir in sorted(condition_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    run_id = run_dir.name
                    index[dataset][model][condition][run_id] = audit_run(run_dir)

    return index


def compute_global_health(index: dict) -> dict:
    """Summarize health across all runs."""
    total_runs = 0
    healthy_runs = 0
    issues = []

    for dataset, models in index.items():
        for model, conditions in models.items():
            for condition, runs in conditions.items():
                for run_id, run_data in runs.items():
                    total_runs += 1
                    if run_data["health"]["ok"]:
                        healthy_runs += 1
                    else:
                        for issue in run_data["health"]["issues"]:
                            issues.append(
                                f"{dataset}/{model}/{condition}/{run_id}: {issue}"
                            )

    return {
        "total_runs": total_runs,
        "healthy_runs": healthy_runs,
        "unhealthy_runs": total_runs - healthy_runs,
        "issues": issues,
    }


@click.command()
@click.argument(
    "results_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for progress.json. Defaults to <results_dir>/progress.json.",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Print issues to stdout after writing."
)
def index(results_dir: Path, output: Path | None, verbose: bool):
    """
    Walk RESULTS_DIR and build a progress.json index file.

    Structure: dataset -> model -> condition -> run_timestamp -> audit data.
    Overwrites any existing progress.json.
    """
    output_path = output or results_dir / "progress.json"

    click.echo(f"Scanning {results_dir} ...")
    index = build_index(results_dir)
    health = compute_global_health(index)

    progress = {
        "generated_at": datetime.now().isoformat(),
        "results_dir": str(results_dir),
        "global_health": health,
        "index": index,
    }

    with open(output_path, "w") as f:
        json.dump(progress, f, indent=2)

    click.echo(f"Written to {output_path}")
    click.echo(
        f"Runs: {health['total_runs']} total, "
        f"{health['healthy_runs']} healthy, "
        f"{health['unhealthy_runs']} unhealthy."
    )

    if verbose and health["issues"]:
        click.echo("\nIssues:")
        for issue in health["issues"]:
            click.echo(f"  ✗ {issue}")
    elif verbose:
        click.echo("No issues found.")