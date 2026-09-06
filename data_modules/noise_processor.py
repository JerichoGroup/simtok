"""Apply configurable thermal noise models to simulation videos."""

from __future__ import annotations

import logging
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from config import get_config

from data_modules.noise_models import (
    config,
    NoiseModel,
    ThermalBlurNoise,
    FixedPatternNoise,
    GaussianNoise,
    TemporalNoise,
    HotPixelsNoise,
    SensorDriftNoise,
    DeadPixelsNoise,
    AGCNoise,
    LowResolutionNoise,
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
        model_factories = {
            "enable_blur": lambda: ThermalBlurNoise(),
            "enable_fixed_pattern": lambda: FixedPatternNoise(width, height),
            "enable_gaussian_noise": lambda: GaussianNoise(),
            "enable_temporal_noise": lambda: TemporalNoise(),
            "enable_low_resolution": lambda: LowResolutionNoise(),
            "enable_agc": lambda: AGCNoise(),
            "enable_sensor_drift": lambda: SensorDriftNoise(),
            "enable_hot_pixels": lambda: HotPixelsNoise(width, height),
            "enable_dead_pixels": lambda: DeadPixelsNoise(width, height),
        }

        return [factory() for toggle, factory in model_factories.items() if config[toggle]]

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
                logger.info("Processed %d frames", frame_count)

        reader.release()
        writer.release()

        logger.info("Done. Saved to %s", output_video)


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


def _process_video_worker(input_video: str, output_video: str) -> str:
    """Process a single video in its own process and return the output path.

    Defined at module level so it is picklable under any multiprocessing
    start method (fork or spawn). It builds its own pipeline; only path
    strings cross the process boundary.
    """
    ThermalVideoPipeline().run(input_video, output_video)
    return output_video


class NoiseProcessor:
    """Apply thermal noise to all videos in a dataset directory."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """Initialize the processor, prioritizing explicit args over TOML config."""
        cfg = get_config()
        paths = cfg.paths
        noise_cfg = cfg.noise

        resolved_root = data_root if data_root is not None else Path(paths["data_root"])

        self.video_directory: Path = resolved_root / "videos"
        self.output_directory: Path = resolved_root / "noise_videos"
        self._max_workers: int = (
            max_workers if max_workers is not None else int(noise_cfg.get("max_workers", 0))
        )

    def process_dataset(self) -> None:
        """Apply thermal noise to every video in the input directory, in parallel."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        video_files = find_video_files(self.video_directory)

        jobs = [
            (
                str(video_file),
                str(build_output_video_path(self.output_directory, video_file)),
            )
            for video_file in video_files
        ]

        worker_count = self._max_workers or min(len(jobs), os.cpu_count() or 1)
        logger.info("Processing %d videos across %d workers", len(jobs), worker_count)

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_input = {
                executor.submit(_process_video_worker, input_video, output_video): input_video
                for input_video, output_video in jobs
            }

            for future in as_completed(future_to_input):
                input_video = future_to_input[future]
                future.result()  # re-raise any worker exception
                logger.info("Finished %s", Path(input_video).name)
