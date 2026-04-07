# tests/test_lanenet2.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """Raw BGR uint8 image, typical camera frame size."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

@pytest.fixture
def dummy_input(dummy_image):
    """LaneNet2Input wrapping the dummy image."""
    inp = MagicMock()
    inp.image = dummy_image
    return inp

@pytest.fixture
def dummy_binary_seg():
    """Fake binary segmentation output — shape (1, 256, 512), values in {0,1}."""
    return np.random.randint(0, 2, (1, 256, 512), dtype=np.uint8)

@pytest.fixture
def dummy_instance_seg():
    """Fake instance embedding output — shape (1, 256, 512, 4)."""
    return np.random.rand(1, 256, 512, 4).astype(np.float32)

@pytest.fixture
def model(dummy_binary_seg, dummy_instance_seg):
    """
    LaneNet2 with _load_model patched out so no real checkpoint is needed.
    sess.run() is stubbed to return fake segmentation tensors.
    """
    with patch("microlane.models.lanenet2.tf"):
        from microlane.models.lanenet2 import LaneNet2

        with patch.object(LaneNet2, "_load_model", return_value=None):
            m = LaneNet2(
                weights_path="fake/weights",
                checkpoint_path="fake/checkpoint",
            )

        # Wire up a fake session
        fake_sess = MagicMock()
        fake_sess.run.return_value = [dummy_binary_seg, dummy_instance_seg]
        setattr(m, "_sess", fake_sess)

        # Fake tensors (just need to be non-None)
        setattr(m, "_input_tensor", MagicMock(name="input_tensor:0"))
        setattr(m, "_binary_output", MagicMock(name="binary_output:0"))
        setattr(m, "_instance_output", MagicMock(name="instance_output:0"))

        return m


# ── _load_model ───────────────────────────────────────────────────────────────

class TestLoadModel:

    def test_called_on_init(self):
        """_load_model must be invoked during __init__."""
        with patch("microlane.models.lanenet2.tf"):
            from microlane.models.lanenet2 import LaneNet2
            with patch.object(LaneNet2, "_load_model") as mock_load:
                LaneNet2(weights_path="w", checkpoint_path="c")
                mock_load.assert_called_once()

    def test_stores_checkpoint_path(self):
        with patch("microlane.models.lanenet2.tf"):
            from microlane.models.lanenet2 import LaneNet2
            with patch.object(LaneNet2, "_load_model", return_value=None):
                m = LaneNet2(weights_path="w", checkpoint_path="my/ckpt")
                assert m.checkpoint_path == "my/ckpt"

    def test_stores_weights_path(self):
        with patch("microlane.models.lanenet2.tf"):
            from microlane.models.lanenet2 import LaneNet2
            with patch.object(LaneNet2, "_load_model", return_value=None):
                m = LaneNet2(weights_path="my/weights", checkpoint_path="c")
                assert m.weights_path == "my/weights"


# ── infer ─────────────────────────────────────────────────────────────────────

class TestInfer:

    def test_returns_lane_prediction(self, model, dummy_input):
        from microlane.schema.prediction import LanePrediction
        result = model.infer(dummy_input)
        assert isinstance(result, LanePrediction)

    def test_sess_run_called_once(self, model, dummy_input):
        model.infer(dummy_input)
        model._sess.run.assert_called_once()

    def test_feed_dict_contains_input_tensor(self, model, dummy_input):
        model.infer(dummy_input)
        call_kwargs = model._sess.run.call_args
        feed_dict = call_kwargs[1].get("feed_dict") or call_kwargs[0][1]
        assert model._input_tensor in feed_dict

    def test_preprocessed_shape_is_correct(self, model, dummy_input):
        """The array fed to sess.run must be (1, 256, 512, 3)."""
        model.infer(dummy_input)
        call_kwargs = model._sess.run.call_args
        feed_dict = call_kwargs[1].get("feed_dict") or call_kwargs[0][1]
        fed_array = feed_dict[model._input_tensor]
        assert fed_array.shape == (1, 256, 512, 3)

    def test_preprocessed_values_in_expected_range(self, model, dummy_input):
        """Normalization: / 127.5 - 1.0 maps [0,255] → [-1, 1]."""
        model.infer(dummy_input)
        call_kwargs = model._sess.run.call_args
        feed_dict = call_kwargs[1].get("feed_dict") or call_kwargs[0][1]
        fed_array = feed_dict[model._input_tensor]
        assert fed_array.min() >= -1.0 - 1e-5
        assert fed_array.max() <=  1.0 + 1e-5

    def test_binary_mask_shape(self, model, dummy_input):
        result = model.infer(dummy_input)
        assert result.binary_mask.shape == (256, 512)  # batch dim dropped

    def test_instance_embeddings_shape(self, model, dummy_input):
        result = model.infer(dummy_input)
        assert result.instance_embeddings.shape == (256, 512, 4)  # batch dim dropped

    def test_binary_mask_values_are_binary(self, model, dummy_input):
        result = model.infer(dummy_input)
        unique_vals = np.unique(result.binary_mask)
        assert set(unique_vals).issubset({0, 1})

    def test_raises_on_none_input(self, model):
        with pytest.raises((AttributeError, TypeError)):
            model.infer(None)


# ── batch_infer ───────────────────────────────────────────────────────────────

class TestBatchInfer:

    def test_returns_list(self, model, dummy_input):
        results = model.batch_infer([dummy_input, dummy_input])
        assert isinstance(results, list)

    def test_output_length_matches_input(self, model, dummy_input):
        batch = [dummy_input] * 5
        results = model.batch_infer(batch)
        assert len(results) == 5

    def test_each_element_is_lane_prediction(self, model, dummy_input):
        from microlane.schema.prediction import LanePrediction
        results = model.batch_infer([dummy_input, dummy_input])
        assert all(isinstance(r, LanePrediction) for r in results)

    def test_empty_batch_returns_empty_list(self, model):
        assert model.batch_infer([]) == []

    def test_infer_called_per_item(self, model, dummy_input):
        """batch_infer must delegate to infer() once per item."""
        with patch.object(model, "infer", wraps=model.infer) as mock_infer:
            model.batch_infer([dummy_input, dummy_input, dummy_input])
            assert mock_infer.call_count == 3