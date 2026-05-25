import click
import pandas as pd

from scripts.core.search_evaluation import search_records


DATASETS      = ["tusimple", "modified_microlane", "microlane"]
MODELS        = ["ufld", "lanenet", "rld_a", "rld_b"]
AUGMENTATIONS = ["normal", "motion_blur", "camera_shake", "lighting_b", "lighting_d"]

PCT  = lambda x: round(x * 100, 2)
MS   = lambda x: round(x * 1000, 2)
R    = lambda x: round(x, 2)


@click.command()
@click.option('--path', '-p', required=True, help='Path to the evaluation csv file')
@click.option('--csv',  '-c', required=True, help='Path to the output summary CSV file')
def summarize(path: str, csv: str):

    df   = pd.read_csv(path)
    rows = []

    for dataset in DATASETS:
        for model in MODELS:
            for augmentation in AUGMENTATIONS:

                records = search_records(df, dataset=dataset, model=model, augmentation=augmentation)

                if records is None or records.empty:
                    continue

                group = records

                row = {
                    "dataset":       dataset,
                    "model":         model,
                    "augmentation":  augmentation,
                    "sample_count":  len(group),

                    "mean_IOU":      PCT(group["IOU"].mean()),
                    "mean_accuracy": PCT(group["accuracy"].mean()),
                    "mean_run_time": MS(group["run_time"].mean()),
                    "mean_fn":       PCT(group["fn"].mean()),
                    "mean_fp":       PCT(group["fp"].mean()),

                    "std_IOU":       PCT(group["IOU"].std()),
                    "std_accuracy":  PCT(group["accuracy"].std()),
                    "std_run_time":  MS(group["run_time"].std()),

                    "min_IOU":       PCT(group["IOU"].min()),
                    "max_IOU":       PCT(group["IOU"].max()),
                    "min_accuracy":  PCT(group["accuracy"].min()),
                    "max_accuracy":  PCT(group["accuracy"].max()),
                    "min_run_time":  MS(group["run_time"].min()),
                    "max_run_time":  MS(group["run_time"].max()),

                    "total_fn":      R(group["fn"].sum()),
                    "total_fp":      R(group["fp"].sum()),
                }

                rows.append(row)

                print(
                    f"[{dataset} | {model} | {augmentation}] "
                    f"n={row['sample_count']}, "
                    f"IOU={row['mean_IOU']}±{row['std_IOU']}%, "
                    f"Acc={row['mean_accuracy']}±{row['std_accuracy']}%, "
                    f"Time={row['mean_run_time']}ms, "
                    f"FN={row['total_fn']}, FP={row['total_fp']}"
                )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv, index=False)

    print(f"\nSummary written to {csv} ({len(summary_df)} rows)")