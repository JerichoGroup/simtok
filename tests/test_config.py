"""Unit tests for config.SimtokConfig — the TOML section-mapping layer.

These tests verify that every section accessor maps the underlying TOML onto
the expected Python structure:
    - flat [section] tables -> dicts (with sub-tables stripped from the parent)
    - nested [section.sub] tables -> their own dicts
    - [[array.of.tables]] -> lists of dicts
    - derived views (noise_flat) and dotted get() access
    - the get_config() singleton behaviour

A self-contained TOML fixture is written to a temp file so the expected values
are explicit and independent of edits to the shipped simtok_config.toml. A
separate test loads the real shipped config to confirm it parses and exposes
all sections.
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable without a conftest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config as config_module
from config import SimtokConfig, get_config, DEFAULT_CONFIG_PATH


# ----------------------------------------------------------------------
# Fixture TOML — exercises every section shape the parser maps
# ----------------------------------------------------------------------

FIXTURE_TOML = """
[paths]
data_root = "data"
usd_path = "usd/maps/scenes/cube.usda"

[collect]
num_samples = 2
video_fps = 20
camera_settle_time_sec = 2.5

[collect.random]
enabled = false
num_povs = 5
backward_m_range = [10.0, 400.0]
right_m_range = [-50.0, 50.0]
up_m_range = [0.0, 100.0]
zoom_range = [0.0, 1.0]

[collect.target]
lat = 32.12345
lon = 35.12345
alt = 0.0

[[collect.povs]]
id = 1
zoom = 0.5

[[collect.povs]]
id = 2
zoom = 0.0

[noise]
gaussian_noise_std = 2.0
blur_sigma = 0.8

[noise.toggles]
enable_blur = true
enable_agc = false

[noise.image]
contrast = 1.0
brightness = 0

[grayscale]
target_name = "Cube"
alpha = 0.001

[grayscale.oscillation]
min = -0.26
max = 0.26

[[grayscale.prims]]
path = "/bboxes/Cube"
default_gray = 0.5

[[grayscale.prims]]
path = "/World/line2"
default_gray = 0.7

[zoom]
zoom_topic = "/simtok/zoom"
zoom_curve = "geometric"

[zoom.sensor]
width_mm = 5.57
height_mm = 3.13

[zoom.hfov]
wide_deg = 65.0
tele_deg = 7.2

[zoom.slew]
duration_sec = 4.0
hz = 20.0

[zoom.logging]
log_path = "/tmp/zoom_node.log"
log_every_sec = 5.0
"""


@pytest.fixture
def cfg(tmp_path):
    """A SimtokConfig loaded from the self-contained fixture TOML."""
    toml_file = tmp_path / "fixture_config.toml"
    toml_file.write_text(FIXTURE_TOML)
    return SimtokConfig(toml_file)


# ----------------------------------------------------------------------
# [paths]
# ----------------------------------------------------------------------

def test_paths_section(cfg):
    assert cfg.paths == {
        "data_root": "data",
        "usd_path": "usd/maps/scenes/cube.usda",
    }


# ----------------------------------------------------------------------
# [collect] and its sub-tables
# ----------------------------------------------------------------------

def test_collect_scalars_exclude_subtables(cfg):
    """collect returns only scalar keys; target, povs and random are stripped out."""
    assert cfg.collect == {
        "num_samples": 2,
        "video_fps": 20,
        "camera_settle_time_sec": 2.5,
    }
    assert "target" not in cfg.collect
    assert "povs" not in cfg.collect
    assert "random" not in cfg.collect


def test_collect_target_section(cfg):
    assert cfg.collect_target == {"lat": 32.12345, "lon": 35.12345, "alt": 0.0}


def test_collect_random_section(cfg):
    assert cfg.collect_random == {
        "enabled": False,
        "num_povs": 5,
        "backward_m_range": [10.0, 400.0],
        "right_m_range": [-50.0, 50.0],
        "up_m_range": [0.0, 100.0],
        "zoom_range": [0.0, 1.0],
    }


def test_collect_povs_array_of_tables(cfg):
    povs = cfg.collect_povs
    assert isinstance(povs, list)
    assert len(povs) == 2
    assert povs[0] == {"id": 1, "zoom": 0.5}
    assert povs[1] == {"id": 2, "zoom": 0.0}


# ----------------------------------------------------------------------
# [noise] and its sub-tables + derived flat view
# ----------------------------------------------------------------------

def test_noise_scalars_exclude_subtables(cfg):
    assert cfg.noise == {"gaussian_noise_std": 2.0, "blur_sigma": 0.8}
    assert "toggles" not in cfg.noise
    assert "image" not in cfg.noise


def test_noise_toggles_section(cfg):
    assert cfg.noise_toggles == {"enable_blur": True, "enable_agc": False}


def test_noise_image_section(cfg):
    assert cfg.noise_image == {"contrast": 1.0, "brightness": 0}


def test_noise_flat_merges_all_noise_subsections(cfg):
    """noise_flat is the union of noise scalars + toggles + image."""
    assert cfg.noise_flat == {
        "gaussian_noise_std": 2.0,
        "blur_sigma": 0.8,
        "enable_blur": True,
        "enable_agc": False,
        "contrast": 1.0,
        "brightness": 0,
    }


# ----------------------------------------------------------------------
# [grayscale] and its sub-tables
# ----------------------------------------------------------------------

def test_grayscale_scalars_exclude_subtables(cfg):
    assert cfg.grayscale == {"target_name": "Cube", "alpha": 0.001}
    assert "oscillation" not in cfg.grayscale
    assert "prims" not in cfg.grayscale


def test_grayscale_oscillation_section(cfg):
    assert cfg.grayscale_oscillation == {"min": -0.26, "max": 0.26}


def test_grayscale_prims_array_of_tables(cfg):
    prims = cfg.grayscale_prims
    assert isinstance(prims, list)
    assert prims == [
        {"path": "/bboxes/Cube", "default_gray": 0.5},
        {"path": "/World/line2", "default_gray": 0.7},
    ]


# ----------------------------------------------------------------------
# [zoom] and its four sub-tables
# ----------------------------------------------------------------------

def test_zoom_scalars_exclude_all_subtables(cfg):
    assert cfg.zoom == {"zoom_topic": "/simtok/zoom", "zoom_curve": "geometric"}
    for sub in ("sensor", "hfov", "slew", "logging"):
        assert sub not in cfg.zoom


def test_zoom_sensor_section(cfg):
    assert cfg.zoom_sensor == {"width_mm": 5.57, "height_mm": 3.13}


def test_zoom_hfov_section(cfg):
    assert cfg.zoom_hfov == {"wide_deg": 65.0, "tele_deg": 7.2}


def test_zoom_slew_section(cfg):
    assert cfg.zoom_slew == {"duration_sec": 4.0, "hz": 20.0}


def test_zoom_logging_section(cfg):
    assert cfg.zoom_logging == {"log_path": "/tmp/zoom_node.log", "log_every_sec": 5.0}


# ----------------------------------------------------------------------
# Section accessors return copies (immutability of internal data)
# ----------------------------------------------------------------------

def test_section_accessors_return_copies(cfg):
    """Mutating a returned dict must not corrupt the config's internal state."""
    paths = cfg.paths
    paths["data_root"] = "MUTATED"
    assert cfg.paths["data_root"] == "data"


# ----------------------------------------------------------------------
# Generic dotted get()
# ----------------------------------------------------------------------

def test_get_dotted_scalar(cfg):
    assert cfg.get("collect.target.lat") == 32.12345
    assert cfg.get("zoom.hfov.wide_deg") == 65.0


def test_get_dotted_returns_table(cfg):
    assert cfg.get("noise.image") == {"contrast": 1.0, "brightness": 0}


def test_get_missing_key_returns_default(cfg):
    assert cfg.get("does.not.exist") is None
    assert cfg.get("does.not.exist", default="fallback") == "fallback"


def test_get_descend_into_scalar_returns_default(cfg):
    """Descending past a scalar (not a table) yields the default, not an error."""
    assert cfg.get("paths.data_root.nope", default="d") == "d"


# ----------------------------------------------------------------------
# get_config() singleton
# ----------------------------------------------------------------------

def test_get_config_is_cached_singleton(tmp_path, monkeypatch):
    """Repeated get_config() calls with no path return the same instance."""
    toml_file = tmp_path / "singleton.toml"
    toml_file.write_text(FIXTURE_TOML)
    # Reset the module-level cache and point the default at our fixture.
    monkeypatch.setattr(config_module, "_config", None)
    first = get_config(toml_file)
    second = get_config()
    assert first is second


def test_get_config_reloads_when_path_given(tmp_path, monkeypatch):
    """Passing an explicit path forces a fresh load (new instance)."""
    toml_file = tmp_path / "reload.toml"
    toml_file.write_text(FIXTURE_TOML)
    monkeypatch.setattr(config_module, "_config", None)
    first = get_config(toml_file)
    second = get_config(toml_file)
    assert first is not second


# ----------------------------------------------------------------------
# The shipped default config must parse and expose every section
# ----------------------------------------------------------------------

def test_shipped_config_parses_and_exposes_all_sections():
    """simtok_config.toml loads and every mapped section is non-empty."""
    assert DEFAULT_CONFIG_PATH.exists()
    real = SimtokConfig(DEFAULT_CONFIG_PATH)

    # Flat/nested dict sections are present and populated.
    for section in (
        real.paths,
        real.collect,
        real.collect_target,
        real.collect_random,
        real.noise,
        real.noise_toggles,
        real.noise_image,
        real.noise_flat,
        real.grayscale,
        real.grayscale_oscillation,
        real.zoom,
        real.zoom_sensor,
        real.zoom_hfov,
        real.zoom_slew,
        real.zoom_logging,
    ):
        assert isinstance(section, dict) and len(section) > 0

    # Array-of-table sections are non-empty lists of dicts.
    for array_section in (real.collect_povs, real.grayscale_prims):
        assert isinstance(array_section, list) and len(array_section) > 0
        assert all(isinstance(entry, dict) for entry in array_section)
