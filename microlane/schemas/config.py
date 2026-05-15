from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class Experiment:
    experiment_directory: Path
    testing_directory: Path
    sample_length: int
    inference_image_sampling_number: int
    model: str
    dataset: str
    augmentation: str

@dataclass
class Dataset:
    name: str
    path: Path
    annotation_file: Path
    dimensions: List[int]

@dataclass
class Datasets:
    tusimple: Dataset
    microlane: Dataset
    modified_microlane: Dataset

@dataclass
class Colors:
    lane_colors: List[List[int]]
    
@dataclass
class Constants:
    h_samples: List[int]
    default_port: int
    colors: Colors

@dataclass
class AugmentationRanges:
    motion_blur_range: tuple[float, float]
    lighting_range: tuple[float, float]
    blur_range: tuple[float, float]
    zoom_range: tuple[float, float]
    rotation_range: tuple[float, float]
    shake_rotation_range: tuple[float, float]
    shake_motion_blur_range: tuple[float, float]

@dataclass
class AugmentationPreset:
    blur: float
    rotation: float
    zoom: float
    lighting: float
    motion_blur: float
    shake: bool

@dataclass
class Augmentation:
    ranges: AugmentationRanges
    presets: dict[str, AugmentationPreset] = field(default_factory=dict)
    
@dataclass
class Model:
    name: str
    container_folder: Path
    image_name: str
    port: int

@dataclass
class Models:
    lanenet: Model
    ufld: Model
    rld_a: Model
    rld_b: Model
    
@dataclass
class Config:
    experiment: Experiment
    constants: Constants
    datasets: Datasets
    models: Models
    augmentation: Augmentation