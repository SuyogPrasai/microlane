from dataclasses import dataclass, field
from pathlib import Path


## Data

@dataclass
class DatasetConfig:
    name: str
    path: Path
    annotation_file: Path
    enabled: bool = True

@dataclass
class DatasetsConfig:
    tusimple: DatasetConfig
    custom_dataset: DatasetConfig

@dataclass
class AugmentationRangesConfig:
    motion_blur_range: tuple[float, float]
    lighting_range: tuple[float, float]
    blur_range: tuple[float, float]
    zoom_range: tuple[float, float]
    rotation_range: tuple[float, float]

@dataclass
class AugmentationPreset:
    blur: float
    rotation: float
    zoom: float
    lighting: float
    motion_blur: float
    shake: bool

    def __getitem__(self, key: str) -> float | bool:
        return getattr(self, key)

@dataclass
class AugmentationConfig:
    ranges: AugmentationRangesConfig
    presets: dict[str, AugmentationPreset] = field(default_factory=dict)
    
 
@dataclass
class DataConfig:
    datasets: DatasetsConfig
    augmentation: AugmentationConfig


## Models

@dataclass
class ModelConfig:
    name: str
    container_folder: Path
    image_name: str
    port: int
    enabled: bool = True

@dataclass
class ModelsConfig:
    lanenet: ModelConfig
    ultra_fast_lane_detection: ModelConfig


## Pipeline

@dataclass
class PipelineConfig:
    name: str
    version: str
    description: str
    default_port: int = 8000


## Experiment

@dataclass
class ExperimentConfig:
    experiment_directory: Path
    testing_directory: Path
    sample_length: int
    inference_image_sampling_number: int


## Root
@dataclass
class Config:
    pipeline: PipelineConfig
    data: DataConfig
    models: ModelsConfig
    experiment: ExperimentConfig