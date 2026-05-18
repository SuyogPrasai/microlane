import click
import pandas as pd


DATASETS      = ["tusimple", "modified_microlane", "microlane"]
MODELS        = ["ufld", "lanenet", "rld_a", "rld_b"]
AUGMENTATIONS = ["normal", "motion_blur", "camera_shake", "lighting_b", "lighting_d"]

IDENTITY_COLS   = ["dataset", "model", "augmentation", "sample_count"]
DEFAULT_METRICS = ["mean_IOU", "mean_accuracy", "mean_fn", "mean_fp", "mean_run_time"]

ALL_METRICS = [
    "mean_IOU", "mean_accuracy", "mean_run_time", "mean_fn", "mean_fp",
    "std_IOU",  "std_accuracy",  "std_run_time",
    "min_IOU",  "max_IOU",       "min_accuracy",  "max_accuracy",
    "min_run_time", "max_run_time", "total_fn", "total_fp",
]

COL_LABELS = {
    "dataset":        "Dataset",
    "model":          "Model",
    "augmentation":   "Augmentation",
    "sample_count":   "Samples",
    "mean_IOU":       "IOU",
    "mean_accuracy":  "Acc",
    "mean_fn":        "FN",
    "mean_fp":        "FP",
    "mean_run_time":  "Time (ms)",
    "std_IOU":        "σ IOU",
    "std_accuracy":   "σ Acc",
    "std_run_time":   "σ Time",
    "min_IOU":        "min IOU",
    "max_IOU":        "max IOU",
    "min_accuracy":   "min Acc",
    "max_accuracy":   "max Acc",
    "min_run_time":   "min Time",
    "max_run_time":   "max Time",
    "total_fn":       "Total FN",
    "total_fp":       "Total FP",
}

METRIC_COLS = set(ALL_METRICS)

RATIO_COLS = {
    "mean_IOU", "mean_accuracy", "std_IOU", "std_accuracy",
    "min_IOU", "max_IOU", "min_accuracy", "max_accuracy",
}

TIME_COLS = {"mean_run_time", "std_run_time", "min_run_time", "max_run_time"}

SORT_ALIASES = {
    "IOU":      "mean_IOU",
    "Accuracy": "mean_accuracy",
    "Acc":      "mean_accuracy",
    "FN":       "mean_fn",
    "FP":       "mean_fp",
    "Time":     "mean_run_time",
}


def fmt(col: str, val) -> str:

    if val == "" or val is None:
        return "—"

    try:
        if col in RATIO_COLS:
            return f"{float(val):.4f}"

        if col in TIME_COLS:
            return f"{float(val):.2f}"

    except (TypeError, ValueError):
        pass

    return str(val)


def label(col: str) -> str:
    return COL_LABELS.get(col, col)


def tag(lbl: str, value: str, color: str = "cyan"):

    click.echo(
        f"  {click.style(f' {lbl} ', bg=color, fg='black', bold=True)}"
        f"{click.style(f' {value}', fg='white')}"
    )


def resolve_sort(sort: str) -> str | None:

    if sort is None:
        return None

    if sort in ALL_METRICS:
        return sort

    if sort in SORT_ALIASES:
        return SORT_ALIASES[sort]

    lower = sort.lower()

    for alias, col in SORT_ALIASES.items():
        if alias.lower() == lower:
            return col

    for col in ALL_METRICS:
        if col.lower() == lower:
            return col

    return None


def print_table(rows: list[dict], columns: list[str], sort_col: str | None = None):

    if not rows:
        return

    widths = {col: len(label(col)) for col in columns}

    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(fmt(col, row.get(col, ""))))

    def hline(left, mid, right):
        return left + mid.join("─" * (widths[c] + 2) for c in columns) + right

    def row_line(cells, header=False):

        parts = []

        for col, cell in zip(columns, cells):

            padded = cell.ljust(widths[col])

            if header:
                fg     = "yellow" if col == sort_col else "white"
                padded = click.style(padded, bold=True, fg=fg)

            elif col in METRIC_COLS:
                fg     = "yellow" if col == sort_col else "cyan"
                padded = click.style(padded, fg=fg)

            else:
                padded = click.style(padded, fg="bright_white")

            parts.append(f" {padded} ")

        pipe = click.style("│", fg="bright_black")

        return pipe + pipe.join(parts) + pipe

    def dim(s):
        return click.style(s, fg="bright_black")

    click.echo()
    click.echo(f"  {dim(hline('╭', '┬', '╮'))}")
    click.echo(f"  {row_line([label(c) for c in columns], header=True)}")
    click.echo(f"  {dim(hline('├', '┼', '┤'))}")

    for row in rows:
        click.echo(f"  {row_line([fmt(col, row.get(col, '')) for col in columns])}")

    click.echo(f"  {dim(hline('╰', '┴', '╯'))}")
    click.echo()


@click.command()
@click.option('--path',         '-p', required=True, help='Path to the summary CSV file.',  metavar='FILE')
@click.option('--dataset',      '-d', default=None,  help='Filter by dataset.',              metavar='NAME')
@click.option('--model',        '-m', default=None,  help='Filter by model.',                metavar='NAME')
@click.option('--augmentation', '-a', default=None,  help='Filter by augmentation.',         metavar='NAME')
@click.option('--metric',             multiple=True, help='Extra metric column(s) to show.', metavar='METRIC')
@click.option('--sort',               default='Accuracy', metavar='METRIC',
              help='Sort largest → smallest. Accepts column names or aliases: IOU, Accuracy/Acc, FN, FP, Time.')
def compare(path: str, dataset: str, model: str, augmentation: str, metric: tuple, sort: str):
    """Compare lane detection model results from a summary CSV."""

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        click.echo(click.style(f"\n  ✗  File not found: {path}\n", fg="red", bold=True))
        raise SystemExit(1)

    active = [(k, v) for k, v in [("dataset", dataset), ("model", model), ("augmentation", augmentation)] if v]

    for k, v in active:
        df = df[df[k] == v]

    if df.empty:
        click.echo(click.style("  ✗  No records match the given filters.\n", fg="red", bold=True))
        raise SystemExit(0)

    invalid = [m for m in metric if m not in ALL_METRICS]

    if invalid:
        click.echo()
        for m in invalid:
            click.echo(click.style(f"  ⚠  Unknown metric ignored: {m}", fg="yellow"))
        click.echo()

    extra   = [m for m in metric if m in ALL_METRICS and m not in DEFAULT_METRICS]
    columns = [c for c in IDENTITY_COLS + DEFAULT_METRICS + extra if c in df.columns]

    sort_col = resolve_sort(sort)

    if sort is not None and sort_col is None:
        click.echo(click.style(f"  ⚠  Unknown sort metric ignored: '{sort}'\n", fg="yellow"))

    if sort_col is not None:

        if sort_col not in df.columns:
            click.echo(click.style(f"  ⚠  Sort column '{sort_col}' not present in CSV — ignored.\n", fg="yellow"))
            sort_col = None

        else:
            df = df.sort_values(sort_col, ascending=False)

            if sort_col not in columns:
                columns.append(sort_col)

            click.echo()

    rows = df[columns].to_dict(orient="records")

    print_table(rows, columns, sort_col=sort_col)

    click.echo(
        f"  {click.style(str(len(rows)), fg='green', bold=True)}"
        f"{click.style(' combination(s) found.\n', fg='bright_black')}"
    )