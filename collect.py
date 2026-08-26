"""Data collection orchestrator for SimTok.

Launches Isaac Sim, moves the camera through configured POVs,
and records video, pose, and bounding-box data for each sample.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Optional

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Empty

import isaac_core_dev_kit.dev_utils as core_utils
from isaac_core_dev_kit.core_capture.bbox_capture import BboxCapture
from isaac_core_dev_kit.core_capture.pose_capture import PoseCapture
from isaac_core_dev_kit.core_capture.video_capture import VideoCapture
from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.udp_bot import UdpBot

from config import get_config


SAMPLE_PATTERN = re.compile(r"sample_(\d{3})_pov_\d+\.mp4")


@dataclass(frozen=True)
class PovConfig:
    """A single camera point-of-view configuration."""

    id: int
    lat: float
    lon: float
    alt: float
    move_up_m: float = 0.0
    pitch_deg: float = 0.0


class DataCollector:
    """Orchestrate data collection from Isaac Sim.

    Resolves parameters using the priority: constructor arg > TOML config.
    If a constructor argument is None, the value is read from the TOML.
    """

    def __init__(
        self,
        *,
        num_samples: Optional[int] = None,
        video_duration_sec: Optional[int] = None,
        video_fps: Optional[int] = None,
        data_root: Optional[Path] = None,
        usd_path: Optional[str] = None,
        povs: Optional[list[PovConfig]] = None,
    ) -> None:
        """Initialize the collector, resolving args vs TOML config."""
        cfg = get_config()
        paths = cfg.paths
        collect = cfg.collect
        target = cfg.collect_target

        # Resolve each parameter: explicit arg wins, else TOML value
        self._num_samples: int = num_samples if num_samples is not None else collect["num_samples"]
        self._video_duration_sec: int = video_duration_sec if video_duration_sec is not None else collect["video_duration_sec"]
        self._video_fps: int = video_fps if video_fps is not None else collect["video_fps"]
        self._usd_path: str = usd_path if usd_path is not None else paths["usd_path"]

        self._initial_scene_load_time_sec: int = collect["initial_scene_load_time_sec"]
        self._camera_settle_time_sec: int = collect["camera_settle_time_sec"]
        self._vertical_move_settle_time_sec: float = collect["vertical_move_settle_time_sec"]

        # Paths
        root = Path(data_root) if data_root is not None else Path(paths["data_root"])
        self._video_dir: Path = root / "videos"
        self._pose_dir: Path = root / "poses"
        self._bbox_dir: Path = root / "bboxes"

        # Target
        self._target_lat: float = target["lat"]
        self._target_lon: float = target["lon"]
        self._target_alt: float = target["alt"]
        self._default_roll: float = target["roll"]
        self._default_pitch: float = target["pitch"]
        self._default_yaw: float = target["yaw"]

        # POVs
        if povs is not None:
            self._povs = povs
        else:
            self._povs = [
                PovConfig(
                    id=p["id"],
                    lat=p["lat"],
                    lon=p["lon"],
                    alt=p["alt"],
                    move_up_m=p.get("move_up_m", 0.0),
                    pitch_deg=p.get("pitch_deg", 0.0),
                )
                for p in cfg.collect_povs
            ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full data-collection run."""
        self._create_output_directories()
        core_utils.safe_rclpy_init()

        oscillation_node, reset_pub, new_pub = self._create_oscillation_publishers()

        sim = HostIsaacManager(
            usd_path=self._usd_path,
            com_udp=True,
            bbox_publisher=True,
            sat=True,
            core_path=".",
            show_isaac_logs=False,
        )

        camera = UdpBot(
            self._target_lat,
            self._target_lon,
            self._target_alt,
            self._default_roll,
            self._default_pitch,
            self._default_yaw,
        )

        video = VideoCapture()
        pose = PoseCapture()
        bbox = BboxCapture()

        with sim:
            sleep(self._initial_scene_load_time_sec)

            video.spin()
            pose.spin()
            bbox.spin()
            sleep(1.0)

            while (
                reset_pub.get_subscription_count() == 0
                or new_pub.get_subscription_count() == 0
            ):
                print("Waiting for thermal node...")
                sleep(0.1)

            self._generate_dataset(camera, video, pose, bbox, reset_pub, new_pub)

        video.shutdown()
        pose.shutdown()
        bbox.shutdown()
        oscillation_node.destroy_node()
        core_utils.safe_rclpy_shutdown()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_output_directories(self) -> None:
        for directory in (self._video_dir, self._pose_dir, self._bbox_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _get_next_sample_id(self) -> int:
        sample_ids = []
        for video in self._video_dir.glob("sample_*_pov_*.mp4"):
            match = SAMPLE_PATTERN.match(video.name)
            if match:
                sample_ids.append(int(match.group(1)))
        return (max(sample_ids) + 1) if sample_ids else 0

    def _build_filename_prefix(self, sample_id: int, pov: PovConfig) -> str:
        return f"sample_{sample_id:03d}_pov_{pov.id}"

    def _move_camera(self, camera: UdpBot, pov: PovConfig) -> None:
        camera.move_to_point(
            pov.lat, pov.lon, pov.alt,
            self._default_roll, self._default_pitch, self._default_yaw,
            look_at_target=False,
        )
        camera.turn_to_point(self._target_lat, self._target_lon, self._target_alt)

        if pov.move_up_m:
            camera.move_up_down(pov.move_up_m)
            sleep(self._vertical_move_settle_time_sec)

        if pov.pitch_deg:
            camera.turn_pitch(pov.pitch_deg)

        sleep(self._camera_settle_time_sec)

    @staticmethod
    def _publish_oscillation(publisher) -> None:
        msg = Empty()
        for _ in range(2):
            publisher.publish(msg)
            sleep(0.05)
        sleep(0.2)

    def _start_recording(self, video, pose, bbox) -> None:
        video.start_capture()
        pose.start_capture()
        bbox.start_capture()

    def _stop_recording(self, video, pose, bbox) -> None:
        bbox.stop_capture()
        pose.stop_capture()
        video.stop_capture()

    def _save_recording(self, sample_id: int, pov: PovConfig, video, pose, bbox) -> None:
        prefix = self._build_filename_prefix(sample_id, pov)
        video.save_data_to(str(self._video_dir / f"{prefix}.mp4"), self._video_fps)
        pose.save_data_to(str(self._pose_dir / f"{prefix}.pkl"))
        bbox.save_data_to(str(self._bbox_dir / f"{prefix}.pkl"))

    def _capture_sample_from_pov(
        self, sample_id, pov, camera, video, pose, bbox, reset_pub
    ) -> None:
        print(f"Recording dataset sample {sample_id:03d} | POV {pov.id}")
        self._move_camera(camera, pov)
        self._publish_oscillation(reset_pub)
        self._start_recording(video, pose, bbox)
        sleep(self._video_duration_sec)
        self._stop_recording(video, pose, bbox)
        self._save_recording(sample_id, pov, video, pose, bbox)

    def _generate_dataset(self, camera, video, pose, bbox, reset_pub, new_pub) -> None:
        start_sample = self._get_next_sample_id()
        end_sample = start_sample + self._num_samples

        for sample_id in range(start_sample, end_sample):
            print("=" * 60)
            print(f"Generating dataset sample {sample_id:03d}")
            print("=" * 60)

            self._publish_oscillation(new_pub)

            for pov in self._povs:
                self._capture_sample_from_pov(
                    sample_id, pov, camera, video, pose, bbox, reset_pub
                )

    @staticmethod
    def _create_oscillation_publishers():
        cfg = get_config()
        grayscale = cfg.grayscale

        node = rclpy.create_node("oscillation_control_publisher")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reset_pub = node.create_publisher(Empty, grayscale["reset_oscillation_topic"], qos)
        new_pub = node.create_publisher(Empty, grayscale["new_oscillation_topic"], qos)
        return node, reset_pub, new_pub
