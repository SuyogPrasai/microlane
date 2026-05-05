import yaml
from pathlib import Path
from dataclasses import fields
from typing import Dict
from typing import get_type_hints, get_origin, get_args

from microlane.schemas.config import Config

def _from_dict(cls, data: dict) -> Config:
    if data is None:
        return None

    hints = get_type_hints(cls)
    kwargs = {}

    for f in fields(cls):
        value = data.get(f.name)
        ftype = hints[f.name]
        origin = get_origin(ftype)

        # Nested dataclass
        if hasattr(ftype, '__dataclass_fields__') and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)

        # tuple[float, float]  e.g. augmentation ranges
        elif origin is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)

        # list[str]
        elif origin is list and isinstance(value, list):
            kwargs[f.name] = value

        # Path fields
        elif ftype is Path or (hasattr(ftype, '__origin__') is False and ftype == Path):
            kwargs[f.name] = Path(value) if value is not None else value

        else:
            kwargs[f.name] = value

    return cls(**kwargs)


def load_config(config_path: Path = Path("/home/suyog/desktop/projects/microlane/configs/config.yaml")) -> Config:
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    if not config_path.suffix == ".yaml":
        raise ValueError(f"Expected .yaml file, got: {config_path.suffix}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return _from_dict(Config, config)