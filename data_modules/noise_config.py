"""Store shared configuration state for the noise processor."""

import tomli as tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH: Path = Path(__file__).parent / "noise_processor_config.toml"

config: dict[str, Any] = {}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Load configuration from a TOML file into the config dict."""
    with open(config_path, "rb") as f:
        toml_data: dict[str, Any] = tomllib.load(f)

    noise: dict[str, Any] = toml_data.get("noise", {})
    toggles: dict[str, Any] = toml_data.get("toggles", {})
    image: dict[str, Any] = toml_data.get("image", {})

    config.update(noise)
    config.update(toggles)
    config.update(image)
    config.update(toml_data.get("general", {}))
