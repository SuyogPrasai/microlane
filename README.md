# Microlane

Evaluating how well modern lane detection models hold up when deployed on small-scale 1/10 RC cars.

This is the code for a pipeline that runs evaluations for different lane detection models across different datasets and environmental conditions (filters).

---

## Documentation

### Core Schemas

There are three main schemas central to this project:

1. **Sample** — The common structure representing a single annotated image in a dataset.
2. **Prediction** — The common output structure of every model evaluation; contains samples + output values.
3. **Evaluation** — The common evaluation structure; contains prediction objects + computed metrics.

The pipeline computes **Evaluation** objects for a subset of samples from a given dataset, across different models and filters.

---

### Key Concepts

#### Filters

Each filter is a pixel-level modification to the image, changing the image tensor itself by adding effects such as motion blur, rotation, zoom, etc. These represent real-world environmental conditions.

#### Augmentors

Augmentors are functions that take a `Sample` object as input and produce a modified `Sample` object as output. This is the implmentation of a `filter`

- They modify the image tensors of the sample.
- They append information about the modification to the `augmentation` property of the sample.

#### Datasets

Datasets are class definitions that represent every type of dataset used in the pipeline. Their primary responsibility is to interact with datasets stored on the local machine and translate them into `Sample` objects that can be fed into the pipeline.

| Dataset | Format |
|---|---|
| TuSimple | JSON Lines (`.json`) |
| CULane | Unknown format |
| Custom (RC car) | XML, CVAT for Image 1.1 |

> The `pathlib` library is used consistently throughout the project for all path operations.


#### Models

Models are class definitions that are a sub class of the Model super class, which contains the code for `prediction` mechanism of the model.

The Model Super class handles things like creating and deploying containers for the specific models.


---

### Data Flow

#### Sample → Prediction

A `Sample` (or set of samples) is passed to a model object, which runs inference and applies preprocessing to produce a `Prediction` object.

#### Prediction → Evaluation

The `Prediction` object is evaluated against metrics such as accuracy and F1-score, and the results are wrapped in an `Evaluation` object.

---

### Model Deployment

To run any lane detection model, the original author's code must be executed in its original environment. Docker is used to containerize this code alongside additional pipeline code, and a FastAPI endpoint is exposed so the pipeline can communicate with the container and produce predictions.

**Deployment steps:**
1. Choose the model to use.
2. Start the model in a Docker container.
3. Expose the model via a RESTful (FastAPI) endpoint.
4. Send data to the endpoint; receive inference output; construct the `Prediction` object.

#### ContainerManager

`ContainerManager` is the Docker Engine API interface through which the pipeline interacts with the Docker client on the host machine. It provides low-level primitives:

- `run_container`
- `stop_container`
- `restart_container`
- List running containers and images (with filtering options)

#### model.py

`model.py` is the higher-level interface for interacting with a model running inside a Docker container. The core orchestration logic for managing containers, communicating with the API endpoint, and constructing `Prediction` objects lives here.