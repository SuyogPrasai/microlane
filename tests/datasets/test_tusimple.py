from microlane.datasets.tusimple import TuSimple
import pytest


@pytest.fixture
def dataset():
    return TuSimple(
        folder_path="../data/TuSimple/TUSimple",
        annotation_file_path="../data/TuSimple/test_label_new.json"
    )


def test_init(dataset):
    assert dataset.folder_path == "../data/TuSimple/TUSimple"
    assert dataset.annotation_file_path == "../data/TuSimple/test_label_new.json"


def test_load_returns_list(dataset):
    data = dataset.load()
    assert isinstance(data, list)


def test_load_respects_number_argument(dataset):
    num = 10
    data = dataset.load(number=num)
    assert isinstance(data, list)
    assert len(data) <= num


def test_load_handles_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset = TuSimple(
            folder_path="../data/TuSimple/TUSimple",
            annotation_file_path=str(tmp_path / "non_existent.json")
        )

def test_load_image_not_implemented(dataset):
    with pytest.raises(Exception):
        dataset.load_image(None)