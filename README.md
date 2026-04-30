# Microlane

Evaluating how well modern lane detection models hold up when deployed on small-scale 1/10 RC cars. Models are benchmarked against TuSimple and CULane datasets, then tested on custom RC car footage under real-world edge conditions.

## Models

| Model | Framework | Link |
|---|---|---|
| LaneNet | TensorFlow | [GitHub](https://github.com/MaybeShewill-CV/lanenet-lane-detection) |
| Ultra Fast Lane Detection | PyTorch | — |
| ConvLSTM-based DNN | Custom architecture | — |

## Evaluation Pipeline

A custom model evaluation pipeline runs comparisons across models and datasets. Datasets are normalized into a common `Sample` object so all pipeline modules can process them uniformly.

**Supported filters:** Lighting · Rotation · Zoom · Blur

![Pipeline diagram](image.png)

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd microlane

# 2. Initialize Python version
pyenv install
pyenv local

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Import model weights
# (place weights in the appropriate directory as described in the docs)

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt
```