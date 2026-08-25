"""Store shared configuration state for the noise processor."""

import tomli as tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "noise_processor_config.toml"

config = {}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Load configuration from a TOML file into the config dict."""
    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)

    noise = toml_data.get("noise", {})
    toggles = toml_data.get("toggles", {})
    image = toml_data.get("image", {})

    config.update(noise)
    config.update(toggles)
    config.update(image)
    config.update(toml_data.get("general", {}))
