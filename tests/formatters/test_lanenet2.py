import pytest
import numpy as np
from unittest.mock import patch
from typing import List

from microlane.schema.sample import Sample, LaneLine
from microlane.schema.model_limbs import LaneNet2Input
from microlane.formatters.lanenet2_formatter import LaneNet2Formatter


# Helpers

def make_lane_line(x_coords, y_coords):
    return LaneLine(x_coordinates=x_coords, y_coordinates=y_coords)


def make_sample(lanes: List[LaneLine], image: np.ndarray, path: str = "test/img.jpg"):
    return Sample(
        image_path=path,
        image=image,
        lanes=lanes,
        raw_annotation={},
    )


H_SAMPLES = list(range(160, 720, 10))   # 56 row anchors, mirrors TuSimple format

# A realistic image that matches the original TuSimple resolution
ORIG_H, ORIG_W = 720, 1280
DUMMY_IMAGE = np.random.randint(0, 255, (ORIG_H, ORIG_W, 3), dtype=np.uint8)

# Two lanes with a mix of valid x-coords and -2 sentinels
LANE_0_X = [-2] * 10 + list(range(300, 300 + 46 * 8,  8))   # starts at 300
LANE_1_X = [-2] * 10 + list(range(900, 900 - 46 * 8, -8))   # starts at 900, mirror
# All-invalid lane
ALL_INVALID_X = [-2] * len(H_SAMPLES)


# Fixtures

@pytest.fixture
def formatter():
    return LaneNet2Formatter(target_size=(512, 256))   # (W, H)


@pytest.fixture
def sample_valid():
    lanes = [
        make_lane_line(LANE_0_X[:len(H_SAMPLES)], H_SAMPLES),
        make_lane_line(LANE_1_X[:len(H_SAMPLES)], H_SAMPLES),
    ]
    return make_sample(lanes, image=DUMMY_IMAGE.copy())


@pytest.fixture
def sample_all_invalid():
    lanes = [
        make_lane_line(ALL_INVALID_X, H_SAMPLES),
        make_lane_line(ALL_INVALID_X, H_SAMPLES),
    ]
    return make_sample(lanes, image=DUMMY_IMAGE.copy())


@pytest.fixture
def batch(sample_valid, sample_all_invalid):
    # reuse valid sample twice + all-invalid once
    return [sample_valid, sample_valid, sample_all_invalid]


# format_one — return type & shape contracts

def test_format_one_returns_lanenet2input(formatter, sample_valid):
    result = formatter.format_one(sample_valid)
    assert isinstance(result, LaneNet2Input)


def test_format_one_image_resized_to_target(formatter, sample_valid):
    W, H = formatter.target_size
    result = formatter.format_one(sample_valid)
    assert result.image.shape == (H, W, 3), (
        f"Expected ({H}, {W}, 3), got {result.image.shape}"
    )


def test_format_one_binary_mask_shape_matches_image(formatter, sample_valid):
    W, H = formatter.target_size
    result = formatter.format_one(sample_valid)
    assert result.binary_mask.shape == (H, W), (
        f"Expected ({H}, {W}), got {result.binary_mask.shape}"
    )


def test_format_one_instance_mask_shape_matches_image(formatter, sample_valid):
    W, H = formatter.target_size
    result = formatter.format_one(sample_valid)
    assert result.instance_mask.shape == (H, W), (
        f"Expected ({H}, {W}), got {result.instance_mask.shape}"
    )


def test_format_one_image_preserves_channel_count(formatter, sample_valid):
    result = formatter.format_one(sample_valid)
    assert result.image.ndim == 3 and result.image.shape[2] == 3


# format_one — mask content correctness

def test_format_one_lanes_with_minus2_are_ignored(formatter, sample_valid):
    """Pixels that correspond to x=-2 sentinel must never be foreground."""
    result = formatter.format_one(sample_valid)
    # The top of the mask (rows that map to the first 10 h_samples, all -2)
    # should be entirely zero — no lane drawn there.
    W, H = formatter.target_size
    top_rows = result.binary_mask[: H // 6, :]   # conservative upper slice
    assert top_rows.sum() == 0, "Expected no foreground in rows where x=-2"


def test_format_one_valid_lane_points_appear_in_mask(formatter, sample_valid):
    result = formatter.format_one(sample_valid)
    assert result.binary_mask.sum() > 0, "Expected foreground pixels for valid lane points"


def test_format_one_instance_mask_has_multiple_lane_ids(formatter, sample_valid):
    result = formatter.format_one(sample_valid)
    unique_ids = set(np.unique(result.instance_mask)) - {0}   # 0 = background
    assert len(unique_ids) >= 2, (
        f"Expected at least 2 lane IDs, got {unique_ids}"
    )


def test_format_one_binary_mask_values_are_zero_or_one(formatter, sample_valid):
    result = formatter.format_one(sample_valid)
    unique = set(np.unique(result.binary_mask))
    assert unique.issubset({0, 1}), f"Binary mask contains unexpected values: {unique}"


# format_one — normalisation

def test_format_one_image_is_normalised(formatter, sample_valid):
    result = formatter.format_one(sample_valid)
    assert result.image.dtype in (np.float32, np.float64), (
        "Image should be a float array after normalisation"
    )
    assert result.image.max() <= 1.0 and result.image.min() >= 0.0, (
        "Normalised image values should be in [0, 1]"
    )


# format_one — edge cases

def test_format_one_all_lanes_invalid_mask_is_zero(formatter, sample_all_invalid):
    """When every x-coord is -2, both masks must be entirely zero."""
    result = formatter.format_one(sample_all_invalid)
    assert result.binary_mask.sum() == 0
    assert result.instance_mask.sum() == 0


# batch_format

def test_batch_format_returns_list(formatter, batch):
    result = formatter.batch_format(batch)
    assert isinstance(result, list)


def test_batch_format_length_matches_input(formatter, batch):
    result = formatter.batch_format(batch)
    assert len(result) == len(batch)


def test_batch_format_each_element_is_lanenet2input(formatter, batch):
    result = formatter.batch_format(batch)
    assert all(isinstance(r, LaneNet2Input) for r in result)


def test_batch_format_empty_list(formatter):
    result = formatter.batch_format([])
    assert result == []