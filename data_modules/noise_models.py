"""Define noise model classes for thermal image simulation."""

import cv2
import numpy as np
from abc import ABC, abstractmethod

from noise_config import config


class NoiseModel(ABC):
    """Provide an abstract interface for image noise models."""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        pass


class ThermalBlur(NoiseModel):
    """Apply Gaussian blur to simulate thermal diffusion."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        return cv2.GaussianBlur(image, (0, 0), config["blur_sigma"])


class FixedPatternNoise(NoiseModel):
    """Apply fixed pattern noise with column, row, and pixel components."""

    def __init__(self, width: int, height: int) -> None:
        """Initialize the fixed pattern noise model."""
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
        """Apply the noise effect to an image."""
        self.pattern += np.random.normal(
            0,
            config["fixed_pattern_drift_std"],
            self.pattern.shape,
        )

        return image + self.pattern


class GaussianNoise(NoiseModel):
    """Apply signal-dependent Gaussian noise."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        image = np.clip(image, 0, 255)

        sigma = (
            config["gaussian_noise_std"]
            + (image / 255.0) * config["gaussian_noise_signal_multiplier"]
        )

        noise = np.random.normal(
            0,
            sigma,
        )

        return image + noise


class TemporalNoise(NoiseModel):
    """Apply frame-to-frame temporal noise."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        noise = np.random.normal(
            0,
            config["temporal_noise_std"],
            image.shape,
        )

        return image + noise


class HotPixels(NoiseModel):
    """Simulate hot pixels with fixed bright spots."""

    def __init__(self, width: int, height: int) -> None:
        """Initialize the hot pixels noise model."""
        self.mask: np.ndarray = (
            np.random.rand(height, width)
            < config["hot_pixel_rate"]
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        result = image.copy()
        result[self.mask] += config["hot_pixel_intensity"]

        return result


class SensorDrift(NoiseModel):
    """Simulate slow sensor baseline drift over time."""

    def __init__(self) -> None:
        """Initialize the sensor drift noise model."""
        self.offset: float = 0.0

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        self.offset += np.random.normal(0, config["sensor_drift_std"])

        return image + self.offset


class DeadPixels(NoiseModel):
    """Simulate dead pixels with fixed dark spots."""

    def __init__(self, width: int, height: int) -> None:
        """Initialize the dead pixels noise model."""
        self.mask: np.ndarray = (
            np.random.rand(height, width)
            < config["dead_pixel_rate"]
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        result = image.copy()
        result[self.mask] = 0

        return result


class AGC(NoiseModel):
    """Apply automatic gain control via histogram normalization."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
        return cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )


class LowResolution(NoiseModel):
    """Simulate low resolution via downsample and upsample."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the noise effect to an image."""
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
