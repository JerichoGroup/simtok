"""Load and provide access to the shared SimTok TOML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import tomli as tomllib


DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent / "simtok_config.toml"


class SimtokConfig:
    """Provide immutable typed access to the shared TOML configuration."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Load the TOML configuration file from disk."""
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
        """Return the [collect] section scalars without sub-tables."""
        section = dict(self._data.get("collect", {}))
        section.pop("target", None)
        section.pop("povs", None)
        return section

    @property
    def collect_target(self) -> Dict[str, float]:
        """Return the [collect.target] section."""
        return dict(self._data.get("collect", {}).get("target", {}))

    @property
    def collect_povs(self) -> list[Dict[str, Any]]:
        """Return the [[collect.povs]] array of tables."""
        return list(self._data.get("collect", {}).get("povs", []))

    @property
    def noise(self) -> Dict[str, Any]:
        """Return the [noise] section scalars without sub-tables."""
        section = dict(self._data.get("noise", {}))
        section.pop("toggles", None)
        section.pop("image", None)
        return section

    @property
    def noise_toggles(self) -> Dict[str, bool]:
        """Return the [noise.toggles] section."""
        return dict(self._data.get("noise", {}).get("toggles", {}))

    @property
    def noise_image(self) -> Dict[str, Any]:
        """Return the [noise.image] section."""
        return dict(self._data.get("noise", {}).get("image", {}))

    @property
    def noise_flat(self) -> Dict[str, Any]:
        """Return a flat dict merging noise params, toggles, and image settings."""
        flat: Dict[str, Any] = {}
        flat.update(self.noise)
        flat.update(self.noise_toggles)
        flat.update(self.noise_image)
        return flat

    @property
    def grayscale(self) -> Dict[str, Any]:
        """Return the [grayscale] section scalars without sub-tables."""
        section = dict(self._data.get("grayscale", {}))
        section.pop("oscillation", None)
        section.pop("prims", None)
        return section

    @property
    def grayscale_oscillation(self) -> Dict[str, Any]:
        """Return the [grayscale.oscillation] section."""
        return dict(self._data.get("grayscale", {}).get("oscillation", {}))

    @property
    def grayscale_prims(self) -> list[Dict[str, Any]]:
        """Return the [[grayscale.prims]] array of tables."""
        return list(self._data.get("grayscale", {}).get("prims", []))

    # ------------------------------------------------------------------
    # Generic access
    # ------------------------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value by dotted key path."""
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
    """Return the singleton SimtokConfig instance."""
    global _config
    if _config is None or config_path is not None:
        _config = SimtokConfig(config_path or DEFAULT_CONFIG_PATH)
    return _config
