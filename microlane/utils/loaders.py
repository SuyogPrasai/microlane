
from microlane.datasets.tusimple import TuSimple
from microlane.datasets.custom_dataset import CustomDataset

from microlane.models.lanenet.model import LaneNet
from microlane.models.ufld.model import UFLD
from microlane.utils.load_config import load_config

config = load_config()

def load_dataset(dataset: str, sample_number: int):
    if dataset == "tusimple":
        ds = TuSimple(
            folder_path=config.data.datasets.tusimple.path,
            annotation_file_path=config.data.datasets.tusimple.annotation_file
        )
    elif dataset == "custom_dataset":
        ds = CustomDataset(
            folder_path=config.data.datasets.custom_dataset.path,
            annotation_file_path=config.data.datasets.custom_dataset.annotation_file
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: tusimple, custom_dataset")

    return ds.load(number=sample_number)


def load_model(model: str):
    if model == "lanenet":
        return LaneNet()
    
    elif model == "ufld":
        return UFLD()
    else:
        raise ValueError(f"Unknown model '{model}'. Choose from: lanenet, ufld")