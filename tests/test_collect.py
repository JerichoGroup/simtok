"""Unit tests for collect.DataCollector initialization/resolution logic.

These cover only __init__ resolution (no Isaac/ROS run):
    - backward_m is negated into forward_m for configured POVs
    - explicit povs= arg takes precedence over everything
    - use_random_povs arg overrides the TOML [collect.random].enabled flag
    - TOML enabled is used when the arg is None
    - scalar params, output dirs and target unpacking

The config singleton is monkeypatched to a self-contained SimtokConfig built
from a fixture TOML so the collector reads known values.
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable without a conftest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config as config_module
from config import SimtokConfig
import collect as collect_module
from collect import DataCollector, PovConfig


# A complete fixture TOML covering every key __init__ reads.
FIXTURE_TOML = """
[paths]
data_root = "data"
usd_path = "usd/maps/scenes/cube.usda"

[collect]
num_samples = 3
video_duration_sec = 4
video_fps = 25
initial_scene_load_time_sec = 5
camera_settle_time_sec = 2

[collect.random]
enabled = false
num_povs = 6
backward_m_range = [10.0, 400.0]
right_m_range = [-50.0, 50.0]
up_m_range = [0.0, 100.0]
zoom_range = [0.0, 1.0]

[collect.target]
lat = 32.12345
lon = 35.12345
alt = 0.0
roll = 0.0
pitch = 0.0
yaw = 90.0

[[collect.povs]]
id = 1
backward_m = 10.0
right_m = 5.0
up_m = 0.0
zoom = 0.5

[[collect.povs]]
id = 2
backward_m = 20.0
right_m = -5.0
up_m = 3.0
zoom = 0.0
"""

# A variant with random collection enabled in TOML.
FIXTURE_TOML_RANDOM_ENABLED = FIXTURE_TOML.replace("enabled = false", "enabled = true")


def _install_config(monkeypatch, tmp_path, toml_text: str) -> None:
    """Point the config singleton at a SimtokConfig built from toml_text."""
    toml_file = tmp_path / "collect_fixture.toml"
    toml_file.write_text(toml_text)
    monkeypatch.setattr(config_module, "_config", SimtokConfig(toml_file))


@pytest.fixture
def cfg_default(monkeypatch, tmp_path):
    _install_config(monkeypatch, tmp_path, FIXTURE_TOML)


@pytest.fixture
def cfg_random_enabled(monkeypatch, tmp_path):
    _install_config(monkeypatch, tmp_path, FIXTURE_TOML_RANDOM_ENABLED)


# ----------------------------------------------------------------------
# Scalar params, dirs, target
# ----------------------------------------------------------------------

def test_scalars_from_toml(cfg_default):
    d = DataCollector()
    assert d._num_samples == 3
    assert d._video_duration_sec == 4
    assert d._video_fps == 25
    assert d._usd_path == "usd/maps/scenes/cube.usda"
    assert d._initial_scene_load_time_sec == 5
    assert d._camera_settle_time_sec == 2


def test_scalar_args_override_toml(cfg_default):
    d = DataCollector(num_samples=99, video_duration_sec=7, video_fps=10, usd_path="x.usda")
    assert d._num_samples == 99
    assert d._video_duration_sec == 7
    assert d._video_fps == 10
    assert d._usd_path == "x.usda"


def test_output_dirs_from_default_root(cfg_default):
    d = DataCollector()
    assert d._video_dir == Path("data") / "videos"
    assert d._pose_dir == Path("data") / "poses"
    assert d._bbox_dir == Path("data") / "bboxes"


def test_output_dirs_from_data_root_arg(cfg_default):
    d = DataCollector(data_root=Path("/tmp/ds"))
    assert d._video_dir == Path("/tmp/ds") / "videos"
    assert d._pose_dir == Path("/tmp/ds") / "poses"
    assert d._bbox_dir == Path("/tmp/ds") / "bboxes"


def test_target_unpacked(cfg_default):
    d = DataCollector()
    assert (d._target_lat, d._target_lon, d._target_alt) == (32.12345, 35.12345, 0.0)
    assert (d._default_roll, d._default_pitch, d._default_yaw) == (0.0, 0.0, 90.0)


# ----------------------------------------------------------------------
# Configured POVs: backward_m negation
# ----------------------------------------------------------------------

def test_configured_povs_negate_backward_into_forward(cfg_default):
    d = DataCollector()
    assert d._povs == [
        PovConfig(id=1, forward_m=-10.0, right_m=5.0, up_m=0.0, zoom=0.5),
        PovConfig(id=2, forward_m=-20.0, right_m=-5.0, up_m=3.0, zoom=0.0),
    ]


# ----------------------------------------------------------------------
# POV selection precedence
# ----------------------------------------------------------------------

def test_explicit_povs_take_precedence(cfg_random_enabled):
    """povs= wins even when random is enabled in TOML."""
    explicit = [PovConfig(id=42, forward_m=-1.0)]
    d = DataCollector(povs=explicit)
    assert d._povs is explicit


def test_toml_enabled_selects_random(cfg_random_enabled):
    d = DataCollector()
    assert len(d._povs) == 6  # num_povs from fixture
    assert [p.id for p in d._povs] == [1, 2, 3, 4, 5, 6]


def test_toml_disabled_selects_configured(cfg_default):
    d = DataCollector()
    assert [p.id for p in d._povs] == [1, 2]


def test_arg_true_overrides_toml_disabled(cfg_default):
    d = DataCollector(use_random_povs=True)
    assert len(d._povs) == 6


def test_arg_false_overrides_toml_enabled(cfg_random_enabled):
    d = DataCollector(use_random_povs=False)
    assert [p.id for p in d._povs] == [1, 2]


def test_arg_none_defers_to_toml(cfg_random_enabled):
    d = DataCollector(use_random_povs=None)
    assert len(d._povs) == 6
