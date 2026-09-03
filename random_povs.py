"""Implements a class that generates random povs in range for simtok."""

from random import uniform
from typing import Tuple

from collect import PovConfig


class RandomPovGenerator:
    """Generate random POVs within configured ranges for SimTok."""

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        zoom_range: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the generator.

        x_range is backward distance (positive = behind target), y_range is the
        right offset (negative = left), z_range is height, and zoom_range is the
        zoom value (0.0 wide .. 1.0 tele).
        """
        self._x_range: Tuple[float, float] = x_range
        self._y_range: Tuple[float, float] = y_range
        self._z_range: Tuple[float, float] = z_range
        self._zoom_range: Tuple[float, float] = zoom_range

    def _generate_random_pov(self, pov_id: int) -> PovConfig:
        """Generate a single random POV.

        Backward distance is negated into forward_m to match the convention used
        elsewhere (forward_m < 0 places the camera behind the target).
        """
        return PovConfig(
            id=pov_id,
            forward_m=-uniform(*self._x_range),
            right_m=uniform(*self._y_range),
            up_m=uniform(*self._z_range),
            zoom=uniform(*self._zoom_range),
        )

    def generate(self, num_povs: int) -> list[PovConfig]:
        """Generate a list of num_povs random POVs with sequential ids."""
        return [self._generate_random_pov(pov_id) for pov_id in range(1, num_povs + 1)]
