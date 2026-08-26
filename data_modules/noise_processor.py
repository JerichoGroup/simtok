"""Process videos by applying configurable thermal noise models."""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List

from noise_config import config, load_config

from noise_models import (
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
    """Read frames from a video file."""

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
        """Read and return the next frame from the video."""
        return self.cap.read()

    def release(self) -> None:
        """Release the video capture resource."""
        self.cap.release()


class VideoWriter:
    """Write frames to a video file."""

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
        """Write a frame to the video file."""
        self.writer.write(frame)

    def release(self) -> None:
        """Release the video writer resource."""
        self.writer.release()


class ThermalProcessor:
    """Apply multiple noise models sequentially to a frame."""

    def __init__(self, noise_models: List[NoiseModel]) -> None:
        """Initialize the thermal processor."""
        self._noise_models: List[NoiseModel] = noise_models

    def process_all_noise_models(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame using each noise model in order."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        for noise_model in self._noise_models:
            image = noise_model.process(image)

        image = image * config["contrast"] + config["brightness"]

        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)


class ThermalVideoPipeline:
    """Run the full thermal noise pipeline on a video."""

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
        """Run noise models on each frame and write the output video."""
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
    """Return all video files in the dataset."""

    video_files = sorted(video_directory.glob("*.mp4"))

    if not video_files:
        raise FileNotFoundError(
            f"No videos found in '{video_directory}'."
        )

    return video_files


def build_output_video_path(
    output_directory: Path,
    input_video: Path,
) -> Path:
    """Return the output path for the processed video."""

    return output_directory / input_video.name


class NoiseProcessor:
    """Provide a high-level API for applying thermal noise to videos."""

    def __init__(
        self,
        input_video_dir: Path | None = None,
        output_video_dir: Path | None = None,
    ) -> None:
        """Initialize the processor and load configuration."""
        load_config()
        self.video_directory: Path = input_video_dir or Path(config["input_video_dir"])
        self.output_directory: Path = output_video_dir or Path(config["output_video_dir"])
        self._pipeline = ThermalVideoPipeline()

    def process_dataset(self) -> None:
        """Apply thermal noise to every video in the dataset."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        video_files = find_video_files(self.video_directory)

        for video_file in video_files:
            output_video = build_output_video_path(
                self.output_directory,
                video_file,
            )
            self.process_video(video_file, output_video)

    def process_video(self, input_video: Path, output_video: Path) -> None:
        """Process a single video with thermal noise."""
        print(f"Input : {input_video.name}")
        print(f"Output: {output_video.name}")

        self._pipeline.run(str(input_video), str(output_video))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply realistic thermal noise to every video in a dataset."
    )

    parser.add_argument(
        "--input-video-dir",
        type=Path,
        default=None,
        help="Directory containing input videos (overrides config file).",
    )

    parser.add_argument(
        "--output-video-dir",
        type=Path,
        default=None,
        help="Directory to write processed videos (overrides config file).",
    )

    return parser.parse_args()


def main() -> None:
    """Run the noise processor from the command line."""
    args = parse_args()

    processor = NoiseProcessor(
        input_video_dir=args.input_video_dir,
        output_video_dir=args.output_video_dir,
    )
    processor.process_dataset()


if __name__ == "__main__":
    main()
