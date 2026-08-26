"""Store shared configuration state for the noise processor.

Loads noise parameters from the unified simtok_config.toml and exposes
them as a flat dict for backward compatibility with noise_models.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure the project root is importable (needed when running from data_modules/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_config

config: dict[str, Any] = {}


def load_config() -> None:
    """Populate the config dict from the shared TOML noise section."""
    cfg = get_config()
    config.clear()
    config.update(cfg.noise_flat)
    # Also inject paths for use by NoiseProcessor
    config["input_video_dir"] = cfg.paths.get("video_dir", "data/videos")
    config["output_video_dir"] = cfg.paths.get("noise_video_dir", "data/noise_videos")
