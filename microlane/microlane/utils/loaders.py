
from microlane.datasets.tusimple import TuSimple
from microlane.datasets.raw import Raw
from microlane.models.lanenet2.model import LaneNet2
from microlane.models.ufld.model import UFLD

def load_dataset(dataset: str, config: dict, sample_number: int):
    if dataset == "tusimple":
        ds = TuSimple(
            annotation_file_path=config['data']['datasets']['tusimple']['annotation_file'],
            folder_path=config['data']['datasets']['tusimple']['path']
        )
    elif dataset == "raw":
        ds = Raw(
            folder_path=config['data']['datasets']['raw']['path']
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: tusimple, raw")

    return ds.load(number=sample_number)


def load_model(model: str, config: dict):
    if model == "lanenet":
        return LaneNet2(
            container_folder=config['models']['lanenet2']['container_folder'],
            image_name=config['models']['lanenet2']['image_name']
        )
    elif model == "ufld":
        return UFLD(
            container_folder=config['models']['ultra_fast_lane_detection']['container_folder'],
            image_name=config['models']['ultra_fast_lane_detection']['image_name']
        )
    else:
        raise ValueError(f"Unknown model '{model}'. Choose from: lanenet, ufld")