from random import randint
from typing import Tuple

class RandomPovGenerator:
    """Generate random povs in range for simtok."""
    def __init__(self, x_range: Tuple[int, int], y_range: Tuple[int, int], z_range: Tuple[int, int]):
        """Initialize the random povs generator with x_range being backword, y_range being right, and z_range being height."""
        self._x_range: Tuple[int, int] = x_range
        self._y_range: Tuple[int, int] = y_range
        self._z_range: Tuple[int, int] = z_range

    def _generate_random_pov

