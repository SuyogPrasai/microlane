# Microlane Pipeline

> A modular, containerized ML pipeline for evaluating lane detection models under small-scale (1/10) RC car conditions.
>
> Full details are discussed in the associated paper: [link_to_paper]

> [!IMPORTANT]
> The pipeline evaluates **4 models** across **3 datasets** with **5 augmentation presets**, processing ~**81,000 images** to generate ~**27,000 predictions**.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Configuration](#configuration)
5. [Data Preparation](#data-preparation)
6. [Usage](#usage)
   - [Running Experiments](#1-running-experiments)
   - [Evaluating Results](#2-evaluating-results)
   - [Summarizing Results](#3-summarizing-results)
   - [Analyzing and Graphing](#4-analyzing-and-graphing)
7. [Implemented Models](#implemented-models)
8. [Evaluation Metrics](#evaluation-metrics)
9. [License](#license)

---

## Overview

Microlane uses Docker containerization to compare lane detection models that have differing dependencies and development environments. Each model runs in its own container, exposed via a FastAPI endpoint, keeping environments isolated and reproducible.

This modular architecture makes the pipeline easy to extend — it can evaluate not just lane detection models, but any ML model that can be wrapped behind a FastAPI interface.

---

## Key Features

- **Modular Architecture** — Easily extendable to include new datasets, models, and metrics.
- **Containerized Models** — Each model runs in a dedicated Docker container with a FastAPI interface, ensuring environment isolation and reproducibility.
- **Standardized Evaluation** — Implements the official TuSimple benchmark (Accuracy, FP, FN) and an ego-lane IoU metric.
- **Image Augmentation** — A flexible augmentation module simulates varied driving conditions: lighting changes, motion blur, camera shake, and more.
- **Command-Line Interface** — Streamlined CLI tools for running batch evaluations and aggregating results into a single CSV.
- **Data Processing** — Includes scripts to convert custom CVAT-annotated datasets into TuSimple-compatible format.

---

## Repository Structure

```
├── microlane/            # Core library source code
│   ├── augmentation/     # Image augmentation filters
│   ├── datasets/         # Data loaders for TuSimple and MicroLane
│   ├── evaluation/       # Evaluation logic (TuSimple metrics, IoU)
│   ├── models/           # Wrappers for containerized models
│   ├── schemas/          # Pydantic and dataclass schemas
│   └── utils/            # Utilities for config, Docker, etc.
├── results/              # Output directory for experiments and graphs
├── scripts/              # CLI tools and notebooks
│   ├── commands/         # CLI command implementations
│   ├── core/             # Core logic for script commands
│   ├── graphing.ipynb    # Notebook for data analysis and visualization
│   └── *.py              # Utility scripts for data conversion
├── config.yaml           # Central configuration file
└── pyproject.toml        # Project dependencies and setup
```

---

## Getting Started

### Prerequisites

- **Python 3.12.10** — The Python version used during development.
- [Docker](https://www.docker.com/get-started) — Required for running lane detection model containers.
- [Poetry](https://python-poetry.org/docs/#installation) — Used for dependency management.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suyogprasai/microlane.git
   cd microlane
   ```

2. Set the correct Python version and activate the Poetry environment:
   ```bash
   pyenv local 3.12.10
   eval (poetry env activate)
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

### Configuration

Before running experiments, configure `config.yaml` — the central file for dataset paths, experiment output directories, and model settings.

- Set `path` and `annotation_file` for both `tusimple` and `microlane` datasets to match your local filesystem.
- Set `experiment_directory` under `experiment` to your desired output location.

---

## Data Preparation

This project uses the **TuSimple** dataset and a custom **MicroLane** dataset. The MicroLane dataset is originally annotated in CVAT's XML format and must be converted to TuSimple's JSON-line format before use.

**Step 1 — Convert MicroLane to TuSimple format:**

```bash
poetry run python scripts/microlane_to_tusimple.py \
  --annotations /path/to/your/cvat_annotations.xml \
  --microlane /path/to/your/unmodified_microlane_images \
  --modified /path/to/your/modified_microlane_images \
  --output ./results/normalized_microlane
```

This produces a `normalized_microlane/` directory with resized images and an `annotations.json` file. Update `config.yaml` to point to these generated assets.

**Step 2 — Visualize converted data (optional):**

Verify the conversion by overlaying ground truth lanes on the converted images:

```bash
poetry run python scripts/visualize_converted.py \
  --images ./results/normalized_microlane/microlane \
  --annotations ./results/normalized_microlane/annotations.json \
  --output ./results/visualized_microlane
```

---

## Usage

### 1. Running Experiments

Experiments are run through Jupyter notebooks, which handle data loading, augmentation, and model inference.

| Model Type | Notebook |
|---|---|
| Single-frame (LaneNet, UFLD) | `scripts/inference.ipynb` |
| Sequence-based (RLD-A, RLD-B) | `scripts/sequence_inference.ipynb` |

Before running, edit the configuration block at the top of the notebook to select the desired `MODEL`, `DATASET`, and `AUGMENTATION` preset. The notebook will automatically start the required Docker container and store raw results and visualizations in the configured experiment directory.

### 2. Evaluating Results

After running experiments, process all prediction files into a single CSV:

```bash
microlane evaluate --path /path/to/experiments/root --csv ./results/evaluation.csv
```

This recursively scans the experiment directory, computes metrics for each prediction, and appends results to `evaluation.csv`.

### 3. Summarizing Results

Aggregate the evaluation CSV to get mean and standard deviation across experiment groups:

```bash
microlane summarize --path ./results/evaluation.csv --csv ./results/summary.csv
```

### 4. Analyzing and Graphing

Use `scripts/graphing.ipynb` to load `evaluation.csv` and generate visualizations:

- **Bar charts** — Compare model and augmentation performance.
- **Line graphs** — Cumulative accuracy over samples.
- **Radar charts** — False Negative vs. False Positive rate comparisons.

---

## Implemented Models

| Model | Type | Description |
|---|---|---|
| **LaneNet** | Single-frame | Segmentation-based model using binary and instance segmentation to identify lanes. |
| **UFLD** (Ultra-Fast Lane Detection) | Single-frame | Formulates lane detection as a row-based classification task for high-speed inference. |
| **RLD-A** (UNet-ConvLSTM) | Sequence-based | Uses a UNet backbone with a ConvLSTM layer to leverage temporal information across frames. |
| **RLD-B** (SegNet-ConvLSTM) | Sequence-based | A variant of RLD-A using a SegNet backbone instead of UNet. |

---

## Evaluation Metrics

### TuSimple Benchmark

| Metric | Description |
|---|---|
| **Accuracy** | Average proportion of correctly predicted lane points per image. |
| **FP (False Positive)** | Rate of predicted lanes that do not correspond to any ground-truth lane. |
| **FN (False Negative)** | Rate of ground-truth lanes that were not detected. |

### Ego-Lane IoU

Calculates the Intersection over Union between the polygon formed by the two innermost predicted lanes (the ego-lane) and the corresponding ground-truth polygon. This measures how accurately the drivable area is detected.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.