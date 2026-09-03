"""Unit tests for random_povs.RandomPovGenerator.

Verifies that generated POVs respect the configured ranges, that the backward
distance is negated into forward_m (the convention used across the collector),
that ids are sequential, and that generation is reproducible under a fixed seed.
"""

import random
import sys
from pathlib import Path

import pytest

# Make the project root importable without a conftest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from random_povs import RandomPovGenerator
from collect import PovConfig


BACKWARD_RANGE = (10.0, 400.0)
RIGHT_RANGE = (-50.0, 50.0)
UP_RANGE = (0.0, 100.0)
ZOOM_RANGE = (0.0, 1.0)


@pytest.fixture
def generator():
    return RandomPovGenerator(BACKWARD_RANGE, RIGHT_RANGE, UP_RANGE, ZOOM_RANGE)


# ----------------------------------------------------------------------
# Count and ids
# ----------------------------------------------------------------------

def test_generate_returns_requested_count(generator):
    povs = generator.generate(7)
    assert len(povs) == 7
    assert all(isinstance(p, PovConfig) for p in povs)


def test_generate_ids_are_sequential_from_one(generator):
    povs = generator.generate(5)
    assert [p.id for p in povs] == [1, 2, 3, 4, 5]


def test_generate_zero_returns_empty(generator):
    assert generator.generate(0) == []


# ----------------------------------------------------------------------
# Ranges
# ----------------------------------------------------------------------

def test_fields_within_configured_ranges(generator):
    for p in generator.generate(200):
        # backward is negated into forward_m: forward_m in [-max, -min]
        assert -BACKWARD_RANGE[1] <= p.forward_m <= -BACKWARD_RANGE[0]
        assert RIGHT_RANGE[0] <= p.right_m <= RIGHT_RANGE[1]
        assert UP_RANGE[0] <= p.up_m <= UP_RANGE[1]
        assert ZOOM_RANGE[0] <= p.zoom <= ZOOM_RANGE[1]


def test_backward_is_negated_into_forward_m():
    """A degenerate backward range pins forward_m to the negated constant."""
    gen = RandomPovGenerator((30.0, 30.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    p = gen.generate(1)[0]
    assert p.forward_m == pytest.approx(-30.0)
    assert p.right_m == pytest.approx(0.0)
    assert p.up_m == pytest.approx(0.0)
    assert p.zoom == pytest.approx(0.0)


def test_zoom_range_defaults_to_zero_when_omitted():
    gen = RandomPovGenerator((10.0, 400.0), (-50.0, 50.0), (0.0, 100.0))
    for p in gen.generate(20):
        assert p.zoom == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def test_seeded_generation_is_reproducible(generator):
    random.seed(1234)
    first = generator.generate(5)
    random.seed(1234)
    second = generator.generate(5)
    assert first == second


def test_different_seeds_produce_different_povs(generator):
    random.seed(1)
    a = generator.generate(5)
    random.seed(2)
    b = generator.generate(5)
    assert a != b
