from pathlib import Path
import json
import numpy as np
from typing import List, Dict


def load_results(eval_file: Path) -> Dict:
    """Load evaluation results from a JSON file."""
    with open(eval_file) as f:
        data = json.load(f)

    results = data if isinstance(data, list) else data.get('results', [])

    metrics = ['accuracy', 'fp', 'fn', 'precision', 'recall', 'f1', 'run_time']

    summary = {}

    for metric in metrics:
        values = np.array([r[metric] for r in results])
        summary[metric] = {
            'mean': float(np.mean(values)),
            'q1':   float(np.percentile(values, 25)),
            'q2':   float(np.percentile(values, 50)),
            'q3':   float(np.percentile(values, 75)),
        }

    sorted_by_f1 = sorted(results, key=lambda r: r['f1'])

    output = {
        'summary': summary,
        'top_5_best':  sorted_by_f1[-5:][::-1],
        'top_5_worst': sorted_by_f1[:5],
    }

    out_path = eval_file.parent / 'summary.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"[✓] summary saved to {out_path}")

    return output
