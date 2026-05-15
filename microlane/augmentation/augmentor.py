from microlane.schemas.sample import Sample, Sequence

from microlane.augmentation.filters.blur import blur
from microlane.augmentation.filters.lighting import lighting
from microlane.augmentation.filters.motion_blur import motion_blur
from microlane.augmentation.filters.rotation import rotation
from microlane.augmentation.filters.zoom import zoom

from microlane.utils.load_config import load_config

import random

config = load_config()


class Augmentor:

    def __init__(self) -> None:
        self.presets = config.augmentation.presets
        self.ranges = config.augmentation.ranges

    def apply_preset(self, sample: Sample, preset_name: str) -> Sample:
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(self.presets)}")

        preset = self.presets[preset_name]

        sample = blur(sample, preset.blur)
        sample = lighting(sample, preset.lighting)
        sample = zoom(sample, preset.zoom)

        if preset.shake:
            rot = random.uniform(*self.ranges.shake_rotation_range)
            mb = random.uniform(*self.ranges.shake_motion_blur_range)
            sample = rotation(sample, rot)
            sample = motion_blur(sample, mb)
        else:
            sample = motion_blur(sample, preset.motion_blur)
            sample = rotation(sample, preset.rotation)

        return sample
    
    def apply_preset_to_sequence(self, sequence: Sequence, preset_name: str) -> Sequence:
    
        return Sequence(
            samples=[self.apply_preset(sample, preset_name) for sample in sequence.samples]
        )