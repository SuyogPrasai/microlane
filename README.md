# Microlane Pipeline

// OverView

![Banner](banner.png)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Configuration](#configuration)
5. [Data Preparation](#data-preparation)
6. [Usage](#usage)
   - [Running Experiments](#running-experiments)
   - [Evaluating Results](#evaluating-results)
   - [Summarizing Results](#summarizing-results)
   - [Analyzing and Graphing](#analyzing-and-graphing)
7. [Models, Metrics and Filters](#models-metrics-and-filters)
8. [Contribution and License](#contribution-and-license)

---

## Architecture

## Key Features

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

- ### Prerequisites
   - You Need to have **Python 3.12.10** to run this code. Other Python versions have not been tested on this code. Along with that, you need to install **Docker Desktop** to run containers in your local machine. Also, install **poetry** and **pyenv** for python version and dependency management.

- ### Installation

   1. First, install the [prerequisites](#prerequisites) in you local machine, and clone this repository.

      ```bash
      git clone https://github.com/suyogprasai/microlane.git
      cd microlane
      ```
   2. Then, set the correct Python Version ( 3.12.10 ), and install dependencies after activating the virtual enviornment using poetry
      ```bash
      pyenv local 3.12.10
      eval (poetry env activate)
      poetry install
      ```

- ### Configuration
   - Before running the Experiments, make sure to download all the dataset in your local machine. Have the Experiment Output Directory Structure Ready, and Configure the `config.yaml` ( The main file for all the options that can be set for the experiments )

## Data Preparation

This project uses the **TuSimple** dataset and a custom **Microlane** dataset. The Microlane dataset is originally in the **XML CVAT for Image 1.1** format, so it must converted to TuSimple's JSON-line format before use.

```bash
python scripts/microlane_to_tusimple.py \
 --annotations /home/suyog/assets/datasets/MicroLane/annotations.xml \
 --microlane /home/suyog/assets/datasets/MicroLane/microlane \
 --modified /home/suyog/assets/datasets/MicroLane/modified_microlane
```

This produces a `normalized_microlane/` directory with resized images and an `annotations.json` file. Update `config.yaml` to point to these generated assets. You can also visualize these new files over the new ground truth values using the following command:

```bash
python scripts/visualize_converted.py \
 --images results/normalized_microlane/modified_microlane/ \
 --annotations results/normalized_microlane/annotations.json
```

## Usage

- ### Running Experiments
   - Experiments are run through Jupyter Notebooks, which handle data loading, augmentation, and model inference. There are two main inference notebooks right now `scripts/inference.ipynb` ( takes in single image )and `scripts/sequence_inference.ipynb` ( takes in multiple images)
  

   - Before running, edit the configuration block at the top of the notebook to select the desired `MODEL`, `DATASET`, and `AUGMENTATION` preset. The notebook will automatically start the required Docker container and store raw results and visualizations in the configured experiment directory.
   

   | Model Type | Notebook |
   |---|---|
   | Single-frame (LaneNet, UFLD) | `scripts/inference.ipynb` |
   | Sequence-based (RLD-A, RLD-B) | `scripts/sequence_inference.ipynb` |

   ***Note***: We had to manually change the value for each result that we make, so we had to change the variables about 60 times manually.


- ### Evaluating Results

   - It took us 3 days to run all the experiments on our local machine, after which we had to run evaluations. ( Which means processing all the output produced by a lane detection models and comparing that with the ground truth. Through which we generate different types of metrics like accuracy and IOU)


   ```bash
   microlane evaluate \
   -p /home/suyog/desktop/projects/microlane/results/experiment \
   -c /home/suyog/desktop/projects/microlane/results/experiment/evaluate.csv
   ```
   - The above command recursively scans the `Experiment` directory, processing all `prediction.json` files to compute metrics for each prediction and consolidate results into a single `evaluate.csv` file.





- ### Summarizing Results
   - Then, we need to create a summary by looking into each experiment group, and creating summary data like averages, standard deviation, quartiles, etc.

   ```bash
   microlane summarize \
   -p /home/suyog/desktop/projects/microlane/results/experiment/evaluate.csv \
   -c /home/suyog/desktop/projects/microlane/results/experiment/summary.csv
   ```


- ### Analyzing and Graphing

   - We mainly use, `graphing.ipynb` and `testing.ipynb` to generate different graphs and visualizations to compare the experiment data. Mainly, we use the following graphs:

      - **Bar charts:** Compare model and augmentation performance.
      - **Line graphs:** Cumulative accuracy over samples.
      - **Radar charts:** False Negative vs. False Positive rate comparisons.


## Models, Metrics and Filters


| Model | Type | Description |
|---|---|---|
| **LaneNet** | Single-frame | Segmentation-based model using binary and instance segmentation. |
| **UFLD** (Ultra-Fast Lane Detection) | Single-frame | Formulates lane detection as row-based classification for high-speed inference. |
| **RLD-A** (UNet-ConvLSTM) | Sequence-based | UNet backbone with ConvLSTM to leverage temporal information across frames. |
| **RLD-B** (SegNet-ConvLSTM) | Sequence-based | Variant of RLD-A using a SegNet backbone instead of UNet. |

| Metric | Source | Description |
|---|---|---|
| **Accuracy** | TuSimple | Proportion of correctly predicted lane points per image. |
| **FP (False Positive)** | TuSimple | Rate of predicted lanes with no matching ground-truth lane. |
| **FN (False Negative)** | TuSimple | Rate of ground-truth lanes that went undetected. |
| **IOU** | Custom | Intersection over Union between the polygon formed by the two innermost predicted lanes (ego-lane) and its ground-truth counterpart. |

| Augmentation | Description |
|---|---|
| **Normal** | No augmentation. |
| **Motion Blur** | Horizontal motion blur using a kernel of size 21. |
| **Camera Shake** | Random rotation (−5° to +5°) combined with random odd-sized horizontal motion blur. |
| **Lighting-B** | Brightness increase of 40% (all RGB channels +40% of 255). |
| **Lighting-D** | Brightness decrease of 40% (all RGB channels −40% of 255). |



## Contribution and License

If you want to contribute to this project, then feel free to fork and add the nescesasry adjustments. This script is designed in a way such that it works not just lane detection models, but  on all kinds of machine learning models we choose. This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.