import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

DARK  = '#0d0d0f'
PANEL = '#16161a'
GRID  = '#2a2a30'

LIGHT_BG    = '#ffffff'
LIGHT_PANEL = '#f5f5f7'
LIGHT_GRID  = '#dddddd'

COLORS_DARK  = ['#00e5ff', '#ff4d6d', '#c8ff00', '#ff9f1c', '#b388ff', '#00ffa3', '#ff6b6b']
COLORS_LIGHT = ['#0077cc', '#e63950', '#5a9e00', '#e07b00', '#7c4dcc', '#00a372', '#cc3333']

METRIC_KEYS = ['accuracy', 'fp', 'fn', 'precision', 'recall', 'f1', 'run_time']


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_results(output_file: Path) -> list[dict]:
    with open(output_file) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('results', [])


def _extract_metrics(results: list[dict]) -> dict[str, list]:
    return {key: [r[key] for r in results] for key in METRIC_KEYS}


# ── Styling ────────────────────────────────────────────────────────────────────

def _apply_style(light_theme: bool = False) -> None:
    if light_theme:
        plt.rcParams.update({
            'figure.facecolor':  LIGHT_BG,
            'axes.facecolor':    LIGHT_PANEL,
            'axes.edgecolor':    LIGHT_GRID,
            'axes.labelcolor':   '#333333',
            'xtick.color':       '#555555',
            'ytick.color':       '#555555',
            'text.color':        '#333333',
            'grid.color':        LIGHT_GRID,
            'grid.linewidth':    0.5,
            'font.family':       'monospace',
            'axes.spines.top':   False,
            'axes.spines.right': False,
        })
    else:
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

def _mean_line(ax, mean_val: float, orientation: str = 'v', light_theme: bool = False) -> None:
    """Draw a dashed mean line and label it."""
    line_color = '#333333' if light_theme else 'white'
    if orientation == 'v':
        ax.axvline(mean_val, color=line_color, linewidth=1.2, linestyle='--', alpha=0.7)
        ax.text(mean_val, ax.get_ylim()[1] * 0.95,
                f' μ={mean_val:.3f}', color=line_color, fontsize=8, va='top')
    else:
        ax.axhline(mean_val, color=line_color, linewidth=0.8, linestyle='--', alpha=0.5)


def _style_axes(ax, title: str, xlabel: str, ylabel: str, color: str) -> None:
    ax.set_title(title, fontsize=13, color=color, pad=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


# ── Plot groups ────────────────────────────────────────────────────────────────

def _save_distributions(metrics: dict, output_folder: Path, light_theme: bool = False) -> None:
    folder = output_folder / 'distribution'
    folder.mkdir(exist_ok=True)
    colors = COLORS_LIGHT if light_theme else COLORS_DARK
    bg = LIGHT_BG if light_theme else DARK

    for i, (metric, values) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=bg)

        ax.hist(values, bins=20, color=color, alpha=0.85, edgecolor=bg, linewidth=0.5)
        _mean_line(ax, float(np.mean(values)), orientation='v', light_theme=light_theme)
        _style_axes(ax,
                    title=f'{metric.upper()} — Distribution',
                    xlabel=metric, ylabel='count', color=color)
        ax.grid(axis='y', alpha=0.4)

        fig.tight_layout()
        fig.savefig(folder / f'{metric}_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[✓] distribution/{metric}_distribution.png")


def _save_progressions(metrics: dict, output_folder: Path, light_theme: bool = False) -> None:
    folder = output_folder / 'progression'
    folder.mkdir(exist_ok=True)
    colors = COLORS_LIGHT if light_theme else COLORS_DARK
    bg = LIGHT_BG if light_theme else DARK
    samples = list(range(1, len(next(iter(metrics.values()))) + 1))

    for i, (metric, values) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=bg)

        ax.plot(samples, values, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(samples, values, alpha=0.08, color=color)

        mean_val = float(np.mean(values))
        _mean_line(ax, mean_val, orientation='h', light_theme=light_theme)
        label_color = '#333333' if light_theme else 'white'
        ax.text(samples[-1], mean_val, f'  μ={mean_val:.3f}',
                color=label_color, fontsize=7, va='center')

        _style_axes(ax,
                    title=f'{metric.upper()} — Per-Sample Progression',
                    xlabel='sample index', ylabel=metric, color=color)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(folder / f'{metric}_progression.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[✓] progression/{metric}_progression.png")


def _save_cumulative(metrics: dict, output_folder: Path, light_theme: bool = False) -> None:
    colors = COLORS_LIGHT if light_theme else COLORS_DARK
    bg = LIGHT_BG if light_theme else DARK
    label_color = '#333333' if light_theme else 'white'
    grid_color = LIGHT_GRID if light_theme else GRID
    samples = list(range(1, len(metrics['accuracy']) + 1))

    acc_vals = np.array(metrics['accuracy'])
    f1_vals  = np.array(metrics['f1'])
    cum_acc  = np.cumsum(acc_vals) / samples
    cum_f1   = np.cumsum(f1_vals)  / samples

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=bg)

    for cum_vals, color, label, va in [
        (cum_acc, colors[0], 'Cumulative Mean Accuracy', 'bottom'),
        (cum_f1,  colors[1], 'Cumulative Mean F1',       'top'),
    ]:
        ax.plot(samples, cum_vals, color=color, linewidth=2, label=label)
        ax.fill_between(samples, cum_vals, alpha=0.07, color=color)
        ax.annotate(f'{cum_vals[-1]:.3f}',
                    xy=(samples[-1], cum_vals[-1]),
                    color=color, fontsize=9, ha='right', va=va)

    ax.set_title('Cumulative Mean — Accuracy & F1', fontsize=14,
                 color=label_color, pad=14, fontweight='bold')
    ax.set_xlabel('samples seen', fontsize=10)
    ax.set_ylabel('cumulative mean', fontsize=10)
    ax.legend(framealpha=0.15, edgecolor=grid_color, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_folder / 'cumulative_mean.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] cumulative_mean.png")


# ── Public entry point ─────────────────────────────────────────────────────────

def draw_evaluation_graphs(output_file: Path, light_theme: bool = False) -> None:
    results = _load_results(output_file)
    metrics = _extract_metrics(results)
    _apply_style(light_theme=light_theme)

    output_folder = output_file.parent
    _save_distributions(metrics, output_folder, light_theme=light_theme)
    _save_progressions(metrics, output_folder, light_theme=light_theme)
    _save_cumulative(metrics, output_folder, light_theme=light_theme)