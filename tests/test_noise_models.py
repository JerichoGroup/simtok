"""Unit tests for data_modules.noise_models.

Noise models are (mostly) stochastic, so we test invariants rather than exact
output:
    - shape preservation (the strongest format guarantee)
    - deterministic behaviors for mask/histogram models
    - statistical properties (seeded) for random models

The pipeline feeds models float32 grayscale frames (see
ThermalProcessor.process_all_noise_models), so tests use float32 input.

noise_models reads `config = get_config().noise_flat` at import time as a
module global; tests that need specific parameter values monkeypatch that dict.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable without a conftest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_modules import noise_models as nm
from data_modules.noise_models import (
    ThermalBlurNoise,
    FixedPatternNoise,
    GaussianNoise,
    TemporalNoise,
    HotPixelsNoise,
    SensorDriftNoise,
    DeadPixelsNoise,
    AGCNoise,
    LowResolutionNoise,
)


WIDTH, HEIGHT = 320, 240


@pytest.fixture
def image():
    """A mid-gray float32 grayscale frame, matching the pipeline's dtype."""
    return np.full((HEIGHT, WIDTH), 100.0, dtype=np.float32)


def _make(cls):
    """Instantiate a model, passing (width, height) to those that require it."""
    if cls in (FixedPatternNoise, HotPixelsNoise, DeadPixelsNoise):
        return cls(WIDTH, HEIGHT)
    return cls()


ALL_MODELS = [
    ThermalBlurNoise,
    FixedPatternNoise,
    GaussianNoise,
    TemporalNoise,
    HotPixelsNoise,
    SensorDriftNoise,
    DeadPixelsNoise,
    AGCNoise,
    LowResolutionNoise,
]


# ----------------------------------------------------------------------
# Structural invariant: shape is preserved by every model
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_MODELS, ids=[c.__name__ for c in ALL_MODELS])
def test_output_shape_preserved(cls, image):
    """Every noise model must return an array of the same shape as its input."""
    model = _make(cls)
    result = model.process(image.copy())
    assert result.shape == image.shape


@pytest.mark.parametrize("cls", ALL_MODELS, ids=[c.__name__ for c in ALL_MODELS])
def test_output_is_ndarray(cls, image):
    """Every model must return a numpy ndarray."""
    model = _make(cls)
    result = model.process(image.copy())
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("cls", ALL_MODELS, ids=[c.__name__ for c in ALL_MODELS])
def test_does_not_mutate_input(cls, image):
    """process() must not mutate the caller's array in place."""
    model = _make(cls)
    original = image.copy()
    model.process(image)
    np.testing.assert_array_equal(image, original)


# ----------------------------------------------------------------------
# Deterministic behavioral invariants: mask models
# ----------------------------------------------------------------------

def test_dead_pixels_zero_masked_locations(image):
    """DeadPixelsNoise sets every masked pixel to exactly 0."""
    model = _make(DeadPixelsNoise)
    result = model.process(image.copy())
    # Wherever the mask is True, the output must be exactly 0.
    assert np.all(result[model.mask] == 0)
    # Wherever the mask is False, the value is unchanged.
    assert np.all(result[~model.mask] == image[~model.mask])


def test_hot_pixels_add_intensity_at_masked_locations(image, monkeypatch):
    """HotPixelsNoise adds exactly hot_pixel_intensity at masked pixels."""
    monkeypatch.setitem(nm.config, "hot_pixel_intensity", 40.0)
    model = _make(HotPixelsNoise)
    result = model.process(image.copy())
    expected = image[model.mask] + 40.0
    np.testing.assert_allclose(result[model.mask], expected)
    # Non-masked pixels unchanged.
    np.testing.assert_array_equal(result[~model.mask], image[~model.mask])


def test_masks_have_frame_shape():
    """Mask-based models build a mask matching (height, width)."""
    for cls in (HotPixelsNoise, DeadPixelsNoise):
        model = cls(WIDTH, HEIGHT)
        assert model.mask.shape == (HEIGHT, WIDTH)


# ----------------------------------------------------------------------
# Deterministic behavioral invariants: histogram / resolution models
# ----------------------------------------------------------------------

def test_agc_normalizes_to_full_range():
    """AGCNoise stretches contrast so min->0 and max->255."""
    # A gradient image so min<max and normalization is observable.
    img = np.linspace(50, 200, WIDTH, dtype=np.float32)
    img = np.tile(img, (HEIGHT, 1))
    result = AGCNoise().process(img)
    assert result.min() == pytest.approx(0, abs=1)
    assert result.max() == pytest.approx(255, abs=1)


def test_agc_on_flat_image_is_stable(image):
    """AGC on a flat image must not blow up (min==max edge case)."""
    result = AGCNoise().process(image.copy())
    assert result.shape == image.shape


def test_low_resolution_is_idempotent_on_shape(image):
    """LowResolutionNoise downsamples then upsamples back to original size."""
    result = LowResolutionNoise().process(image.copy())
    assert result.shape == image.shape


def test_low_resolution_loses_high_frequency_detail():
    """A sharp single-pixel spike should be blurred/spread by down+up sampling."""
    img = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    img[HEIGHT // 2, WIDTH // 2] = 255.0
    result = LowResolutionNoise().process(img)
    # The isolated peak value should drop after the resolution round-trip.
    assert result[HEIGHT // 2, WIDTH // 2] < 255.0


# ----------------------------------------------------------------------
# Statistical invariants: purely random additive models (seeded)
# ----------------------------------------------------------------------

def test_gaussian_noise_is_zero_mean(image, monkeypatch):
    """GaussianNoise adds approximately zero-mean noise (no DC shift)."""
    monkeypatch.setitem(nm.config, "gaussian_noise_std", 5.0)
    monkeypatch.setitem(nm.config, "gaussian_noise_signal_multiplier", 0.0)
    np.random.seed(42)
    result = GaussianNoise().process(image.copy())
    diff = result - image
    # Mean of added noise should be near 0 over many pixels.
    assert abs(float(diff.mean())) < 0.5


def test_temporal_noise_std_matches_config(image, monkeypatch):
    """TemporalNoise adds noise whose std is close to the configured value."""
    monkeypatch.setitem(nm.config, "temporal_noise_std", 3.0)
    np.random.seed(0)
    result = TemporalNoise().process(image.copy())
    diff = result - image
    assert float(diff.std()) == pytest.approx(3.0, rel=0.1)


def test_temporal_noise_is_reproducible_with_seed(image, monkeypatch):
    """Same seed -> identical temporal noise (determinism under fixed RNG)."""
    monkeypatch.setitem(nm.config, "temporal_noise_std", 2.0)
    np.random.seed(123)
    a = TemporalNoise().process(image.copy())
    np.random.seed(123)
    b = TemporalNoise().process(image.copy())
    np.testing.assert_array_equal(a, b)


def test_sensor_drift_accumulates(image, monkeypatch):
    """SensorDriftNoise accumulates a cumulative offset across calls."""
    monkeypatch.setitem(nm.config, "sensor_drift_std", 1.0)
    np.random.seed(7)
    model = SensorDriftNoise()
    model.process(image.copy())
    first_offset = model.offset
    model.process(image.copy())
    second_offset = model.offset
    # Offset changes across frames (drift), i.e. it is stateful.
    assert first_offset != second_offset


def test_fixed_pattern_is_stateful_across_frames(image, monkeypatch):
    """FixedPatternNoise drifts its pattern each frame (temporal correlation)."""
    monkeypatch.setitem(nm.config, "fixed_pattern_drift_std", 0.01)
    np.random.seed(1)
    model = _make(FixedPatternNoise)
    pattern_before = model.pattern.copy()
    model.process(image.copy())
    # The stored pattern drifts after processing.
    assert not np.array_equal(pattern_before, model.pattern)
