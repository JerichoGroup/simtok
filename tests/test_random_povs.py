"""Unit tests for random_povs.RandomPovGenerator.

Verifies that generated POVs respect the configured ranges, that the depth
distance is negated into depth_m (the convention used across the collector),
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


DEPTH_RANGE = (10.0, 400.0)
HORIZONTAL_RANGE = (-50.0, 50.0)
VERTICAL_RANGE = (0.0, 100.0)
ZOOM_RANGE = (0.0, 1.0)


@pytest.fixture
def generator():
    return RandomPovGenerator(DEPTH_RANGE, HORIZONTAL_RANGE, VERTICAL_RANGE, ZOOM_RANGE)


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
        # depth is negated into depth_m: depth_m in [-max, -min]
        assert -DEPTH_RANGE[1] <= p.depth_m <= -DEPTH_RANGE[0]
        assert HORIZONTAL_RANGE[0] <= p.horizontal_m <= HORIZONTAL_RANGE[1]
        assert VERTICAL_RANGE[0] <= p.vertical_m <= VERTICAL_RANGE[1]
        assert ZOOM_RANGE[0] <= p.zoom <= ZOOM_RANGE[1]


def test_depth_is_negated_into_depth_m():
    """A degenerate depth range pins depth_m to the negated constant."""
    gen = RandomPovGenerator((30.0, 30.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    p = gen.generate(1)[0]
    assert p.depth_m == pytest.approx(-30.0)
    assert p.horizontal_m == pytest.approx(0.0)
    assert p.vertical_m == pytest.approx(0.0)
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


# ----------------------------------------------------------------------
# Edge and degenerate inputs
# ----------------------------------------------------------------------

def test_generate_negative_count_returns_empty(generator):
    """A negative count yields range(1, <=0) -> no POVs (documents behavior)."""
    assert generator.generate(-1) == []
    assert generator.generate(-100) == []


def test_inverted_range_still_stays_within_bounds():
    """random.uniform(a, b) with a > b returns a value in [b, a].

    The generator does not sort its ranges, so an inverted depth range must
    still produce depth_m within the negated bounds. This pins the contract.
    """
    gen = RandomPovGenerator((400.0, 10.0), (50.0, -50.0), (100.0, 0.0), (1.0, 0.0))
    for p in gen.generate(200):
        # depth_m = -uniform(400, 10) -> in [-400, -10]
        assert -400.0 <= p.depth_m <= -10.0
        assert -50.0 <= p.horizontal_m <= 50.0
        assert 0.0 <= p.vertical_m <= 100.0
        assert 0.0 <= p.zoom <= 1.0


def test_all_fields_pinned_to_distinct_constants():
    """Degenerate (a, a) ranges pin every field, verifying each mapping at once."""
    gen = RandomPovGenerator((30.0, 30.0), (7.0, 7.0), (12.0, 12.0), (0.4, 0.4))
    p = gen.generate(1)[0]
    assert p.depth_m == pytest.approx(-30.0)  # depth negated into depth_m
    assert p.horizontal_m == pytest.approx(7.0)
    assert p.vertical_m == pytest.approx(12.0)
    assert p.zoom == pytest.approx(0.4)
