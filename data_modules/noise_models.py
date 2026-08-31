"""Define noise model classes for thermal image simulation."""

import cv2
import numpy as np
from abc import ABC, abstractmethod

from config import get_config

config = get_config().noise_flat


class NoiseModel(ABC):
    """Define the abstract interface for all noise models."""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image and return the result."""
        pass


class ThermalBlurNoise(NoiseModel):
    """Simulate thermal diffusion via Gaussian blur."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        return cv2.GaussianBlur(image, (0, 0), config["blur_sigma"])


class FixedPatternNoise(NoiseModel):
    """Simulate sensor fixed-pattern noise with column, row, and pixel components."""

    def __init__(self, width: int, height: int) -> None:
        """Generate the fixed noise pattern for the given frame dimensions."""
        column_noise = np.random.normal(0, config["fixed_pattern_column_std"], width)
        row_noise = np.random.normal(0, config["fixed_pattern_row_std"], height)

        pixel_noise = np.random.normal(
            0,
            config["fixed_pattern_std"],
            (height, width),
        )

        self.pattern: np.ndarray = (
            pixel_noise
            + column_noise[np.newaxis, :]
            + row_noise[:, np.newaxis]
        ).astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Add the drifting fixed pattern to the image."""
        self.pattern += np.random.normal(
            0,
            config["fixed_pattern_drift_std"],
            self.pattern.shape,
        )

        return image + self.pattern


class GaussianNoise(NoiseModel):
    """Simulate signal-dependent Gaussian read noise."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Add signal-dependent Gaussian noise to the image."""
        image = np.clip(image, 0, 255)

        sigma = (
            config["gaussian_noise_std"]
            + (image / 255.0) * config["gaussian_noise_signal_multiplier"]
        )
        noise = np.random.normal(0, sigma)
        return image + noise


class TemporalNoise(NoiseModel):
    """Simulate frame-to-frame temporal noise."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Add random temporal noise to the image."""
        noise = np.random.normal(
            0,
            config["temporal_noise_std"],
            image.shape,
        )

        return image + noise


class HotPixelsNoise(NoiseModel):
    """Simulate stuck-high hot pixels on the sensor."""

    def __init__(self, width: int, height: int) -> None:
        """Generate a random hot pixel mask for the given frame dimensions."""
        self.mask: np.ndarray = (
            np.random.rand(height, width)
            < config["hot_pixel_rate"]
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Add hot pixel intensity at masked locations."""
        result = image.copy()
        result[self.mask] += config["hot_pixel_intensity"]

        return result


class SensorDriftNoise(NoiseModel):
    """Simulate slow sensor baseline drift over time."""

    def __init__(self) -> None:
        """Initialize the cumulative drift offset to zero."""
        self.offset: float = 0.0

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply cumulative drift offset to the image."""
        self.offset += np.random.normal(0, config["sensor_drift_std"])

        return image + self.offset


class DeadPixelsNoise(NoiseModel):
    """Simulate stuck-low dead pixels on the sensor."""

    def __init__(self, width: int, height: int) -> None:
        """Generate a random dead pixel mask for the given frame dimensions."""
        self.mask: np.ndarray = (
            np.random.rand(height, width)
            < config["dead_pixel_rate"]
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Zero out pixel values at masked locations."""
        result = image.copy()
        result[self.mask] = 0

        return result


class AGCNoise(NoiseModel):
    """Simulate automatic gain control via histogram normalization."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Normalize the image histogram to the full 0-255 range."""
        return cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )


class LowResolutionNoise(NoiseModel):
    """Simulate low sensor resolution via downsample and upsample."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Downsample the image by half and upsample back to original size."""
        h, w = image.shape

        small = cv2.resize(
            image,
            (w // 2, h // 2),
            interpolation=cv2.INTER_AREA,
        )

        return cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

