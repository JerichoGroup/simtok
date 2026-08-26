"""Load and provide access to the shared SimTok TOML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import tomli as tomllib


DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent / "simtok_config.toml"


class SimtokConfig:
    """Immutable accessor for the shared TOML configuration.

    Loads the TOML once and exposes typed section accessors.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Load configuration from disk."""
        with open(config_path, "rb") as f:
            self._data: Dict[str, Any] = tomllib.load(f)

    # ------------------------------------------------------------------
    # Section accessors
    # ------------------------------------------------------------------

    @property
    def paths(self) -> Dict[str, str]:
        """Return the [paths] section."""
        return dict(self._data.get("paths", {}))

    @property
    def collect(self) -> Dict[str, Any]:
        """Return the [collect] section (scalars only, no sub-tables)."""
        section = dict(self._data.get("collect", {}))
        section.pop("target", None)
        section.pop("povs", None)
        return section

    @property
    def collect_target(self) -> Dict[str, float]:
        """Return [collect.target]."""
        return dict(self._data.get("collect", {}).get("target", {}))

    @property
    def collect_povs(self) -> list[Dict[str, Any]]:
        """Return [[collect.povs]] array of tables."""
        return list(self._data.get("collect", {}).get("povs", []))

    @property
    def noise(self) -> Dict[str, Any]:
        """Return [noise] section (scalars only)."""
        section = dict(self._data.get("noise", {}))
        section.pop("toggles", None)
        section.pop("image", None)
        return section

    @property
    def noise_toggles(self) -> Dict[str, bool]:
        """Return [noise.toggles]."""
        return dict(self._data.get("noise", {}).get("toggles", {}))

    @property
    def noise_image(self) -> Dict[str, Any]:
        """Return [noise.image]."""
        return dict(self._data.get("noise", {}).get("image", {}))

    @property
    def noise_flat(self) -> Dict[str, Any]:
        """Return a flat dict merging noise params, toggles, and image settings.

        This matches the legacy `config` dict format expected by noise_models.py.
        """
        flat: Dict[str, Any] = {}
        flat.update(self.noise)
        flat.update(self.noise_toggles)
        flat.update(self.noise_image)
        return flat

    # ------------------------------------------------------------------
    # Generic access
    # ------------------------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value by dotted key path (e.g. 'paths.data_root')."""
        keys = dotted_key.split(".")
        node: Any = self._data
        for key in keys:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return default
            if node is None:
                return default
        return node


# Module-level singleton — import and use directly.
_config: SimtokConfig | None = None


def get_config(config_path: Path | None = None) -> SimtokConfig:
    """Return the singleton SimtokConfig instance.

    Call with a path to override the default location (useful for tests).
    """
    global _config
    if _config is None or config_path is not None:
        _config = SimtokConfig(config_path or DEFAULT_CONFIG_PATH)
    return _config
