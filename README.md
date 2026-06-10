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
   - You Need to have **Python 3.12.10** to run this code. Other Python versions have not been tested on this code. Along with that, you need to install **Docker Desktop** to run containers in your local machine. Also, install poetry and **pyenv** for **python** version and dependency management.

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
   - Before running the Experiments, make sure

## Data Preparation

## Usage

- ### Running Experiments

- ### Evaluating Results

- ### Summarizing Results

- ### Analyzing and Graphing

## Models, Metrics and Filters

## Contribution and License

If you want to contribute to this project, then feel free to fork and add the nescesasry adjustments. This script is designed in a way such that it works not just lane detection models, but  on all kinds of machine learning models we choose. This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.