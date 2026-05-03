import click
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

ALL_METRICS = ["accuracy", "f1", "precision", "recall", "fp", "fn", "run_time"]

METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "f1":        "F1 Score",
    "precision": "Precision",
    "recall":    "Recall",
    "fp":        "FPR",
    "fn":        "FNR",
    "run_time":  "Run Time (s)",
}

# ── Multi-value option helper ────────────────────────────────────────────────

def _split_values(ctx, param, value):
    """
    Allow space- or comma-separated values for a multiple=True option.
    e.g. --metric f1 accuracy  OR  --metric f1,accuracy  OR  --metric f1 --metric accuracy
    all produce the same result: ('f1', 'accuracy')
    """
    result = []
    for v in value:
        for part in v.replace(",", " ").split():
            result.append(part)
    return tuple(result)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_progress(progress_path: Path) -> dict:
    with open(progress_path) as f:
        return json.load(f)


def collect_runs(
    progress_data: dict,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    conditions: tuple[str, ...],
) -> list[dict]:
    """
    Flatten the index into a list of run dicts with added scope keys.
    Filters by datasets/models/conditions if provided.
    Only includes runs that have evaluation data.
    """
    runs = []
    index = progress_data.get("index", {})

    for dataset, model_map in index.items():
        if datasets and dataset not in datasets:
            continue
        for model, condition_map in model_map.items():
            if models and model not in models:
                continue
            for condition, run_map in condition_map.items():
                if conditions and condition not in conditions:
                    continue
                for run_id, run_data in run_map.items():
                    if "evaluation" not in run_data:
                        continue
                    runs.append({
                        "dataset":      dataset,
                        "model":        model,
                        "condition":    condition,
                        "run_id":       run_id,
                        "summary":      run_data["evaluation"]["summary"],
                        "sample_count": run_data["evaluation"]["sample_count"],
                    })

    return runs


def format_value(metric: str, value: float) -> str:
    if metric == "run_time":
        return f"{value:.4f}s"
    return f"{value:.4f}"


def print_table(runs: list[dict], metrics: list[str]) -> None:
    """Print a rich terminal table of runs × metrics."""
    if not runs:
        click.echo("No evaluated runs match the given filters.")
        return

    model_w   = max(len(r["model"])     for r in runs) + 2
    cond_w    = max(len(r["condition"]) for r in runs) + 2
    dataset_w = max(len(r["dataset"])   for r in runs) + 2
    metric_w  = 12

    header_parts = [
        "Dataset".ljust(dataset_w),
        "Model".ljust(model_w),
        "Condition".ljust(cond_w),
        "Samples".rjust(8),
    ]
    for m in metrics:
        header_parts.append(METRIC_LABELS.get(m, m).rjust(metric_w))

    sep    = "─" * (dataset_w + model_w + cond_w + 8 + metric_w * len(metrics) + len(metrics) + 4)
    header = "  ".join(header_parts)

    click.echo()
    click.echo(sep)
    click.echo(header)
    click.echo(sep)

    for r in sorted(runs, key=lambda x: (x["dataset"], x["model"], x["condition"])):
        summary    = r["summary"]
        row_parts  = [
            r["dataset"].ljust(dataset_w),
            r["model"].ljust(model_w),
            r["condition"].ljust(cond_w),
            str(r["sample_count"]).rjust(8),
        ]
        for m in metrics:
            val = summary.get(m)
            if val is None:
                row_parts.append("N/A".rjust(metric_w))
            else:
                row_parts.append(format_value(m, val).rjust(metric_w))
        click.echo("  ".join(row_parts))

    click.echo(sep)
    click.echo()


def save_table(runs: list[dict], metrics: list[str], save_path: Path) -> None:
    """Save the runs table to a CSV or Excel file based on the file extension."""
    try:
        import pandas as pd
    except ImportError:
        click.echo("Error: pandas is required for --save. Run: pip install pandas openpyxl")
        return

    rows = []
    for r in sorted(runs, key=lambda x: (x["dataset"], x["model"], x["condition"])):
        row = {
            "dataset":   r["dataset"],
            "model":     r["model"],
            "condition": r["condition"],
            "samples":   r["sample_count"],
        }
        for m in metrics:
            val = r["summary"].get(m)
            row[METRIC_LABELS.get(m, m)] = val
        rows.append(row)

    df     = pd.DataFrame(rows)
    suffix = save_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(save_path, index=False)
        click.echo(f"  Saved CSV → {save_path}")
    elif suffix in (".xlsx", ".xls", ".excel"):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            click.echo("Error: openpyxl is required for Excel export. Run: pip install openpyxl")
            return

        if suffix in (".xls", ".excel"):
            save_path = save_path.with_suffix(".xlsx")
            click.echo(f"  Note: Saving as .xlsx instead of {suffix}")

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
            ws = writer.sheets["Results"]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        click.echo(f"  Saved Excel → {save_path}")
    else:
        click.echo(f"Error: Unsupported file extension '{suffix}'. Use .csv or .xlsx")


# ── Plotting ─────────────────────────────────────────────────────────────────

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def plot_metric(
    runs: list[dict],
    metric: str,
    group_by: str,
    split_by: str,
    save_dir: Path,
) -> None:
    groups   = list(dict.fromkeys(r[group_by] for r in runs))
    splits   = list(dict.fromkeys(r[split_by] for r in runs))
    n_groups = len(groups)
    n_splits = len(splits)
    bar_width = 0.8 / n_splits
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(6, n_groups * n_splits * 1.2), 5))

    for i, split_val in enumerate(splits):
        values = []
        for group_val in groups:
            match = next(
                (r for r in runs if r[group_by] == group_val and r[split_by] == split_val),
                None,
            )
            values.append(match["summary"].get(metric) if match else 0.0)

        offset = (i - n_splits / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=split_val,
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            if val is not None and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)} — by {group_by} / split by {split_by}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(title=split_by.capitalize(), fontsize=8, title_fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    click.echo(f"  Saved → {out}")
    plt.close(fig)


# ── Command ──────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--progress", "-p",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to progress.json",
)
@click.option(
    "--dataset", "-d",
    multiple=True,
    callback=_split_values,
    help="Filter by dataset(s).  Accepts space- or comma-separated values.\n"
         "e.g. --dataset tusimple culane  OR  --dataset tusimple,culane",
)
@click.option(
    "--model", "-m",
    multiple=True,
    callback=_split_values,
    help="Filter by model(s).  Accepts space- or comma-separated values.\n"
         "e.g. --model lanenet2 ufld  OR  --model lanenet2,ufld",
)
@click.option(
    "--condition", "-c",
    multiple=True,
    callback=_split_values,
    help="Filter by condition(s).  Accepts space- or comma-separated values.\n"
         "e.g. --condition normal motion_blur  OR  --condition normal,motion_blur",
)
@click.option(
    "--metric",
    multiple=True,
    callback=_split_values,
    help="Metrics to compare.  Accepts space- or comma-separated values.\n"
         "e.g. --metric f1 accuracy  OR  --metric f1,accuracy  (default: all)",
)
@click.option(
    "--plot",
    type=click.Path(path_type=Path),
    default=None,
    help="Save plots to this directory, organised by dataset/model/condition",
)
@click.option(
    "--save", "-s",
    type=click.Path(path_type=Path),
    default=None,
    help="Save results table to a file. Supports .csv and .xlsx e.g. --save results.csv",
)
@click.pass_context
def compare(
    ctx: click.Context,
    progress: Path,
    dataset: tuple[str, ...],
    model: tuple[str, ...],
    condition: tuple[str, ...],
    metric: tuple[str, ...],
    plot: Path | None,
    save: Path | None,
) -> None:
    """
    Compare evaluated runs from progress.json across models and conditions.

    Examples:\n
      compare --model lanenet2\n
      compare --condition normal motion_blur\n
      compare --model lanenet2 ufld --condition normal camera_shake --metric f1 accuracy\n
      compare --dataset tusimple --plot ./plots\n
      compare --dataset tusimple --save results.csv\n
      compare --dataset tusimple --save results.xlsx
    """
    progress_data = load_progress(progress)

    metrics = list(metric) if metric else ALL_METRICS
    runs    = collect_runs(progress_data, dataset, model, condition)

    if not runs:
        click.echo("No evaluated runs match the given filters.")
        return

    print_table(runs, metrics)

    if save:
        save_table(runs, metrics, save)

    unique_models     = list(dict.fromkeys(r["model"]     for r in runs))
    unique_conditions = list(dict.fromkeys(r["condition"] for r in runs))

    if len(unique_models) == 1:
        group_by, split_by = "condition", "dataset"
    elif len(unique_conditions) == 1:
        group_by, split_by = "model", "dataset"
    else:
        group_by, split_by = "condition", "model"

    if plot:
        unique_datasets  = list(dict.fromkeys(r["dataset"]   for r in runs))
        scope_models     = list(dict.fromkeys(r["model"]     for r in runs))
        scope_conditions = list(dict.fromkeys(r["condition"] for r in runs))

        dataset_part   = unique_datasets[0]  if len(unique_datasets)  == 1 else "all_datasets"
        model_part     = scope_models[0]     if len(scope_models)     == 1 else "all_models"
        condition_part = scope_conditions[0] if len(scope_conditions) == 1 else "all_conditions"

        save_dir = plot / dataset_part / model_part / condition_part

        click.echo(f"Plotting {len(metrics)} metric(s) → {save_dir}")
        click.echo()

        for m in metrics:
            plot_metric(runs, m, group_by, split_by, save_dir)

        click.echo("\nAll plots saved.")