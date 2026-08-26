"""Apply configurable thermal noise models to simulation videos."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from data_modules.noise_config import config, load_config

from data_modules.noise_models import (
    NoiseModel,
    ThermalBlur,
    FixedPatternNoise,
    GaussianNoise,
    TemporalNoise,
    HotPixels,
    SensorDrift,
    DeadPixels,
    AGC,
    LowResolution,
)


class VideoReader:
    """Read frames sequentially from a video file."""

    def __init__(self, filename: str) -> None:
        """Open a video file for reading."""
        self.cap = cv2.VideoCapture(filename)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {filename}")

    @property
    def fps(self) -> float:
        """Return the video frame rate."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Return the video frame width in pixels."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Return the video frame height in pixels."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> tuple[bool, np.ndarray]:
        """Read the next frame from the video."""
        return self.cap.read()

    def release(self) -> None:
        """Release the video capture resource."""
        self.cap.release()


class VideoWriter:
    """Write grayscale frames to a video file."""

    def __init__(self, filename: str, fps: float, width: int, height: int) -> None:
        """Open a video file for writing."""
        self.writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
            False,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame to the video file."""
        self.writer.write(frame)

    def release(self) -> None:
        """Release the video writer resource."""
        self.writer.release()


class ThermalProcessor:
    """Apply a sequence of noise models to produce a thermal frame."""

    def __init__(self, noise_models: List[NoiseModel]) -> None:
        """Initialize the processor with an ordered list of noise models."""
        self._noise_models: List[NoiseModel] = noise_models

    def process_all_noise_models(self, frame: np.ndarray) -> np.ndarray:
        """Convert a BGR frame to grayscale and apply all noise models."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        for noise_model in self._noise_models:
            image = noise_model.process(image)

        image = image * config["contrast"] + config["brightness"]
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)


class ThermalVideoPipeline:
    """Process an entire video through the thermal noise pipeline."""

    def _build_noise_models(self, width: int, height: int) -> List[NoiseModel]:
        """Build the list of active noise models based on config toggles."""
        models: List[NoiseModel] = []

        if config["enable_blur"]:
            models.append(ThermalBlur())
        if config["enable_fixed_pattern"]:
            models.append(FixedPatternNoise(width, height))
        if config["enable_gaussian_noise"]:
            models.append(GaussianNoise())
        if config["enable_temporal_noise"]:
            models.append(TemporalNoise())
        if config["enable_low_resolution"]:
            models.append(LowResolution())
        if config["enable_agc"]:
            models.append(AGC())
        if config["enable_sensor_drift"]:
            models.append(SensorDrift())
        if config["enable_hot_pixels"]:
            models.append(HotPixels(width, height))
        if config["enable_dead_pixels"]:
            models.append(DeadPixels(width, height))

        return models

    def run(self, input_video: str, output_video: str) -> None:
        """Process all frames from input and write the noised output."""
        reader = VideoReader(input_video)

        writer = VideoWriter(
            output_video,
            reader.fps,
            reader.width,
            reader.height,
        )

        noise_models = self._build_noise_models(reader.width, reader.height)
        processor = ThermalProcessor(noise_models)

        frame_count: int = 0

        while True:
            ret, frame = reader.read()

            if not ret:
                break

            output = processor.process_all_noise_models(frame)
            writer.write(output)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

        reader.release()
        writer.release()

        print(f"Done. Saved to {output_video}")


def find_video_files(video_directory: Path) -> list[Path]:
    """Return all mp4 files sorted from the given directory."""
    video_files = sorted(video_directory.glob("*.mp4"))

    if not video_files:
        raise FileNotFoundError(
            f"No videos found in '{video_directory}'."
        )

    return video_files


def build_output_video_path(output_directory: Path, input_video: Path) -> Path:
    """Build the output path by placing the input filename in the output directory."""
    return output_directory / input_video.name


class NoiseProcessor:
    """Apply thermal noise to all videos in a dataset directory."""

    def __init__(
        self,
        input_video_dir: Optional[Path] = None,
        output_video_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the processor, prioritizing explicit args over TOML config."""
        load_config()
        self.video_directory: Path = input_video_dir if input_video_dir is not None else Path(config["input_video_dir"])
        self.output_directory: Path = output_video_dir if output_video_dir is not None else Path(config["output_video_dir"])
        self._pipeline = ThermalVideoPipeline()

    def process_dataset(self) -> None:
        """Apply thermal noise to every video in the input directory."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        video_files = find_video_files(self.video_directory)

        for video_file in video_files:
            output_video = build_output_video_path(
                self.output_directory,
                video_file,
            )
            self.process_video(video_file, output_video)

    def process_video(self, input_video: Path, output_video: Path) -> None:
        """Process a single video file with thermal noise."""
        print(f"Input : {input_video.name}")
        print(f"Output: {output_video.name}")

        self._pipeline.run(str(input_video), str(output_video))
