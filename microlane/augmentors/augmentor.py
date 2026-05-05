from microlane.schemas.sample import Sample

from microlane.augmentors.blur import blur
from microlane.augmentors.lighting import lighting
from microlane.augmentors.motion_blur import motion_blur
from microlane.augmentors.rotation import rotation
from microlane.augmentors.zoom import zoom

from microlane.utils.load_config import load_config

config = load_config()


class Augmentor:

    def __init__(self) -> None:

        self.presets = config.data.augmentation.presets


    def _apply_preset(self, sample: Sample, preset_name: str) -> Sample:

        preset = self.presets[preset_name]

        sample = blur(sample, preset.blur)
        sample = lighting(sample, preset.lighting)
        sample = motion_blur(sample, preset.motion_blur)
        sample = rotation(sample, preset.rotation)
        sample = zoom(sample, preset.zoom)

        return sample


    def normal(self, sample: Sample) -> Sample:
        return self._apply_preset(sample, "normal")

    def lighting_d(self, sample: Sample) -> Sample:
        return self._apply_preset(sample, "lighting_d")

    def lighting_b(self, sample: Sample) -> Sample:
        return self._apply_preset(sample, "lighting_b")

    def motion_blur(self, sample: Sample) -> Sample:
        return self._apply_preset(sample, "motion-blur")

    def camera_shake(self, sample: Sample) -> Sample:
        return self._apply_preset(sample, "camera-shake")