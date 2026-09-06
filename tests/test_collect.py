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
from collect import DataCollector, PovConfig, SAMPLE_PATTERN


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

# A random-enabled variant with zoom_range omitted, to exercise the
# random_cfg.get("zoom_range", (0.0, 0.0)) fallback in _generate_random_povs.
FIXTURE_TOML_RANDOM_NO_ZOOM = FIXTURE_TOML_RANDOM_ENABLED.replace(
    "zoom_range = [0.0, 1.0]\n", ""
)


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


@pytest.fixture
def cfg_random_no_zoom(monkeypatch, tmp_path):
    _install_config(monkeypatch, tmp_path, FIXTURE_TOML_RANDOM_NO_ZOOM)


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


# ----------------------------------------------------------------------
# Random config plumbing: fallbacks, range propagation, edge cases
# ----------------------------------------------------------------------

def test_empty_explicit_povs_list_is_respected(cfg_random_enabled):
    """povs=[] is not None, so it short-circuits random/configured selection."""
    d = DataCollector(povs=[])
    assert d._povs == []


def test_random_zoom_range_omitted_falls_back_to_zero(cfg_random_no_zoom):
    """When [collect.random] omits zoom_range, all generated zooms are 0.0."""
    d = DataCollector()
    assert len(d._povs) == 6
    assert all(p.zoom == 0.0 for p in d._povs)


def test_random_ranges_propagate_from_toml(cfg_random_enabled):
    """Generated POVs respect the TOML ranges, not just the requested count.

    Confirms collect wires backward/right/up/zoom ranges through to the
    generator (backward_m is negated into forward_m).
    """
    import random

    random.seed(2024)
    d = DataCollector()
    for p in d._povs:
        assert -400.0 <= p.forward_m <= -10.0  # backward_m_range [10, 400] negated
        assert -50.0 <= p.right_m <= 50.0       # right_m_range
        assert 0.0 <= p.up_m <= 100.0           # up_m_range
        assert 0.0 <= p.zoom <= 1.0             # zoom_range


def test_random_missing_num_povs_raises_key_error(monkeypatch, tmp_path):
    """enabled=true but num_povs absent -> KeyError from _generate_random_povs."""
    toml_text = FIXTURE_TOML_RANDOM_ENABLED.replace("num_povs = 6\n", "")
    _install_config(monkeypatch, tmp_path, toml_text)
    with pytest.raises(KeyError):
        DataCollector()


def test_random_missing_range_raises_key_error(monkeypatch, tmp_path):
    """enabled=true but a required *_range key absent -> KeyError."""
    toml_text = FIXTURE_TOML_RANDOM_ENABLED.replace(
        "backward_m_range = [10.0, 400.0]\n", ""
    )
    _install_config(monkeypatch, tmp_path, toml_text)
    with pytest.raises(KeyError):
        DataCollector()


# ----------------------------------------------------------------------
# Filename / sample-id logic
# ----------------------------------------------------------------------

def test_build_filename_prefix_zero_pads_sample(cfg_default):
    d = DataCollector()
    prefix = d._build_filename_prefix(3, PovConfig(id=2))
    assert prefix == "sample_003_pov_2"


def test_build_filename_prefix_large_sample(cfg_default):
    d = DataCollector()
    prefix = d._build_filename_prefix(123, PovConfig(id=10))
    assert prefix == "sample_123_pov_10"


def test_next_sample_id_empty_dir_is_zero(cfg_default, tmp_path):
    d = DataCollector(data_root=tmp_path)
    d._video_dir.mkdir(parents=True, exist_ok=True)
    assert d._get_next_sample_id() == 0


def test_next_sample_id_is_max_plus_one(cfg_default, tmp_path):
    d = DataCollector(data_root=tmp_path)
    d._video_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sample_000_pov_1.mp4", "sample_003_pov_1.mp4", "sample_003_pov_2.mp4"):
        (d._video_dir / name).touch()
    assert d._get_next_sample_id() == 4


def test_next_sample_id_ignores_non_matching_files(cfg_default, tmp_path):
    d = DataCollector(data_root=tmp_path)
    d._video_dir.mkdir(parents=True, exist_ok=True)
    (d._video_dir / "sample_005_pov_1.mp4").touch()
    # Non-matching names must be ignored by SAMPLE_PATTERN.
    for name in ("notes.txt", "sample_x_pov_1.mp4", "sample_5_pov_1.mp4", "sample_005_pov_1.pkl"):
        (d._video_dir / name).touch()
    assert d._get_next_sample_id() == 6


def test_sample_pattern_matching():
    """SAMPLE_PATTERN matches the canonical name and rejects malformed ones."""
    assert SAMPLE_PATTERN.match("sample_007_pov_3.mp4").group(1) == "007"
    assert SAMPLE_PATTERN.match("sample_7_pov_3.mp4") is None   # sample not 3 digits
    assert SAMPLE_PATTERN.match("sample_007_pov_.mp4") is None  # missing pov id
    assert SAMPLE_PATTERN.match("sample_007_pov_3.pkl") is None  # wrong extension
