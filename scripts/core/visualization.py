import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

DARK  = '#0d0d0f'
PANEL = '#16161a'
GRID  = '#2a2a30'

COLORS = ['#00e5ff', '#ff4d6d', '#c8ff00', '#ff9f1c', '#b388ff', '#00ffa3', '#ff6b6b']

METRIC_KEYS = ['accuracy', 'fp', 'fn', 'precision', 'recall', 'f1', 'run_time']


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_results(output_file: Path) -> list[dict]:
    with open(output_file) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('results', [])


def _extract_metrics(results: list[dict]) -> dict[str, list]:
    return {key: [r[key] for r in results] for key in METRIC_KEYS}


# ── Styling ────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.rcParams.update({
        'figure.facecolor':  DARK,
        'axes.facecolor':    PANEL,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   '#cccccc',
        'xtick.color':       '#888888',
        'ytick.color':       '#888888',
        'text.color':        '#cccccc',
        'grid.color':        GRID,
        'grid.linewidth':    0.5,
        'font.family':       'monospace',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


# ── Shared plot helpers ────────────────────────────────────────────────────────

def _mean_line(ax, mean_val: float, orientation: str = 'v') -> None:
    """Draw a dashed white mean line and label it."""
    if orientation == 'v':
        ax.axvline(mean_val, color='white', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(mean_val, ax.get_ylim()[1] * 0.95,
                f' μ={mean_val:.3f}', color='white', fontsize=8, va='top')
    else:
        ax.axhline(mean_val, color='white', linewidth=0.8, linestyle='--', alpha=0.5)


def _style_axes(ax, title: str, xlabel: str, ylabel: str, color: str) -> None:
    ax.set_title(title, fontsize=13, color=color, pad=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


# ── Plot groups ────────────────────────────────────────────────────────────────

def _save_distributions(metrics: dict, output_folder: Path) -> None:
    folder = output_folder / 'distribution'
    folder.mkdir(exist_ok=True)

    for i, (metric, values) in enumerate(metrics.items()):
        color = COLORS[i % len(COLORS)]
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=DARK)

        ax.hist(values, bins=20, color=color, alpha=0.85, edgecolor=DARK, linewidth=0.5)
        _mean_line(ax, float(np.mean(values)), orientation='v')
        _style_axes(ax,
                    title=f'{metric.upper()} — Distribution',
                    xlabel=metric, ylabel='count', color=color)
        ax.grid(axis='y', alpha=0.4)

        fig.tight_layout()
        fig.savefig(folder / f'{metric}_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[✓] distribution/{metric}_distribution.png")


def _save_progressions(metrics: dict, output_folder: Path) -> None:
    folder = output_folder / 'progression'
    folder.mkdir(exist_ok=True)
    samples = list(range(1, len(next(iter(metrics.values()))) + 1))

    for i, (metric, values) in enumerate(metrics.items()):
        color = COLORS[i % len(COLORS)]
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=DARK)

        ax.plot(samples, values, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(samples, values, alpha=0.08, color=color)

        mean_val = float(np.mean(values))
        _mean_line(ax, mean_val, orientation='h')
        ax.text(samples[-1], mean_val, f'  μ={mean_val:.3f}',
                color='white', fontsize=7, va='center')

        _style_axes(ax,
                    title=f'{metric.upper()} — Per-Sample Progression',
                    xlabel='sample index', ylabel=metric, color=color)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(folder / f'{metric}_progression.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[✓] progression/{metric}_progression.png")


def _save_cumulative(metrics: dict, output_folder: Path) -> None:
    samples = list(range(1, len(metrics['accuracy']) + 1))

    acc_vals = np.array(metrics['accuracy'])
    f1_vals  = np.array(metrics['f1'])
    cum_acc  = np.cumsum(acc_vals) / samples
    cum_f1   = np.cumsum(f1_vals)  / samples

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)

    for cum_vals, color, label, va in [
        (cum_acc, COLORS[0], 'Cumulative Mean Accuracy', 'bottom'),
        (cum_f1,  COLORS[1], 'Cumulative Mean F1',       'top'),
    ]:
        ax.plot(samples, cum_vals, color=color, linewidth=2, label=label)
        ax.fill_between(samples, cum_vals, alpha=0.07, color=color)
        ax.annotate(f'{cum_vals[-1]:.3f}',
                    xy=(samples[-1], cum_vals[-1]),
                    color=color, fontsize=9, ha='right', va=va)

    ax.set_title('Cumulative Mean — Accuracy & F1', fontsize=14,
                 color='white', pad=14, fontweight='bold')
    ax.set_xlabel('samples seen', fontsize=10)
    ax.set_ylabel('cumulative mean', fontsize=10)
    ax.legend(framealpha=0.15, edgecolor=GRID, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_folder / 'cumulative_mean.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] cumulative_mean.png")


# ── Public entry point ─────────────────────────────────────────────────────────

def draw_evaluation_graphs(output_file: Path) -> None:
    results = _load_results(output_file)
    metrics = _extract_metrics(results)
    _apply_style()

    output_folder = output_file.parent
    _save_distributions(metrics, output_folder)
    _save_progressions(metrics, output_folder)
    _save_cumulative(metrics, output_folder)