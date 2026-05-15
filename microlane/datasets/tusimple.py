# Be able to load sample objects from the locally downloaded TuSimple Dataset
# Source of TuSimple Dataset: https://www.kaggle.com/datasets/manideep1108/tusimple

from pathlib import Path
from typing import List, Optional
import cv2, json
import numpy as np

from microlane.schemas.sample import Sample, Sequence


class TuSimple():

    def __init__(
        self,
        folder_path: Path,
        annotation_file_path: Path
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
                        dataset="TuSimple"
                    )
                )

        return samples

    def load_sequences(self, number: int = 500, sequence_length: int = 5) -> List[Sequence]:

        sequences: List[Sequence] = []

        with open(self.annotation_file_path, 'r') as f:

            for i, line in enumerate(f):
                if i >= number:
                    break

                data = json.loads(line)

                anchor_rel = Path(data['raw_file'])
                anchor_path = self.folder_path / anchor_rel

                if not anchor_path.exists():
                    print(f"Warning: Anchor image not found at [{anchor_path}], skipping sequence.")
                    continue

                anchor_image = self._load_image(anchor_path)

                if anchor_image is None:
                    print(f"Warning: Failed to load anchor image at [{anchor_path}], skipping sequence.")
                    continue

                anchor_frame_num = int(anchor_rel.stem)
                clip_dir = anchor_rel.parent

                sequence_samples: List[Sample] = []
                all_frames_loaded = True

                for offset in range(sequence_length - 1, 0, -1):

                    frame_path = self.folder_path / clip_dir / f"{anchor_frame_num - offset}.jpg"

                    if not frame_path.exists():
                        print(f"Warning: Context frame not found at [{frame_path}], skipping sequence.")
                        all_frames_loaded = False
                        break

                    frame_image = self._load_image(frame_path)

                    if frame_image is None:
                        print(f"Warning: Failed to load context frame at [{frame_path}], skipping sequence.")
                        all_frames_loaded = False
                        break

                    sequence_samples.append(
                        Sample(
                            image_path=str(frame_path),
                            image=frame_image,
                            lanes=np.array([]),
                            h_samples=np.array([]),
                            dataset="TuSimple"
                        )
                    )

                if not all_frames_loaded:
                    continue

                sequence_samples.append(
                    Sample(
                        image_path=str(anchor_path),
                        image=anchor_image,
                        lanes=np.array(data['lanes']),
                        h_samples=np.array(data['h_samples']),
                        dataset="TuSimple"
                    )
                )

                sequences.append(Sequence(samples=sequence_samples))

        return sequences