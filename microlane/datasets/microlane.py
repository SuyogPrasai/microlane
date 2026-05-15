from pathlib import Path
from typing import List, Optional
import cv2, json
import numpy as np
from collections import defaultdict

from microlane.schemas.sample import Sample, Sequence


class MicroLane():

    def __init__(
        self,
        folder_path: Path,
        annotation_file_path: Path,
        ) -> None:

        self.folder_path = folder_path
        self.annotation_file_path = annotation_file_path

        if self.annotation_file_path.exists() is False:
            raise FileNotFoundError(f"Annotation file not found at {self.annotation_file_path}")

        if self.folder_path.exists() is False:
            raise FileNotFoundError(f"Folder path not found at {self.folder_path}")

        if self.folder_path.is_dir() is False:
            raise NotADirectoryError(f"Provided folder path is not a directory: {self.folder_path}")

        if self.annotation_file_path.is_file() is False:
            raise ValueError(f"Provided annotation path is not a file: {self.annotation_file_path}")

        if self.annotation_file_path.suffix.lower() != '.json':
            raise ValueError(f"Annotation file must be a JSON file: {self.annotation_file_path}")


    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:

        image = cv2.imread(str(image_path))

        if image is None:
            return None

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    def load(self, number: int = 500) -> List[Sample]:

        samples: List[Sample] = []

        with open(self.annotation_file_path, 'r') as f:

            for i, line in enumerate(f):
                if i >= number:
                    break

                data = json.loads(line)

                image_path = self.folder_path / data['raw_file']

                if not image_path.exists():
                    print(f"Warning: Image file not found at [{image_path}], skipping sample.")
                    continue

                image = self._load_image(image_path)

                if image is None:
                    print(f"Warning: Failed to load image at [{image_path}], skipping sample.")
                    continue

                samples.append(
                    Sample(
                        image_path=str(image_path),
                        image=image,
                        lanes=np.array(data['lanes']),
                        h_samples=np.array(data['h_samples']),
                        dataset="MicroLane"
                    )
                )

        return samples


    def load_sequences(self, number: int = 500, sequence_length: int = 5) -> List[Sequence]:

        # Group loaded samples by clip folder, preserving frame order
        clips: defaultdict[str, List[Sample]] = defaultdict(list)

        with open(self.annotation_file_path, 'r') as f:

            for i, line in enumerate(f):
                if i >= number:
                    break

                data = json.loads(line)

                image_path = self.folder_path / data['raw_file']

                if not image_path.exists():
                    print(f"Warning: Image file not found at [{image_path}], skipping sample.")
                    continue

                image = self._load_image(image_path)

                if image is None:
                    print(f"Warning: Failed to load image at [{image_path}], skipping sample.")
                    continue

                clip_name = Path(data['raw_file']).parent.name

                clips[clip_name].append(
                    Sample(
                        image_path=str(image_path),
                        image=image,
                        lanes=np.array(data['lanes']),
                        h_samples=np.array(data['h_samples']),
                        dataset="MicroLane"
                    )
                )

        # Sliding window over each clip's frames
        sequences: List[Sequence] = []

        for clip_name, samples in clips.items():

            if len(samples) < sequence_length:
                print(f"Warning: Clip [{clip_name}] has {len(samples)} frames, fewer than sequence length {sequence_length}, skipping.")
                continue

            for start in range(len(samples) - sequence_length + 1):
                sequences.append(Sequence(samples=samples[start : start + sequence_length]))

        return sequences